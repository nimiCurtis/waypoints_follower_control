#!/usr/bin/env python3
from typing import Tuple, List

import numpy as np
import torch

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import message_filters
from zed_interfaces.msg import ObjectsStamped

from pilot_deploy.inference import PilotPlanner, get_inference_config
from pilot_utils.transforms import transform_images, ObservationTransform
from pilot_utils.deploy.deploy_utils import msg_to_pil
from pilot_utils.utils import tic, toc, from_numpy

def clip_angle(theta: float) -> float:
    """
    Clips an angle to the range [-π, π].

    Args:
        theta (float): Input angle in radians.

    Returns:
        float: Clipped angle within the range [-π, π].
    """
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


def create_path_msg(waypoints: List[Tuple], frame_id: str)->Path:
    """
    Creates a ROS Path message from a list of waypoints.

    Args:
        waypoints (list of tuple): List of waypoints, where each waypoint is a tuple (x, y, yaw).
        frame_id (str): The frame of reference for the path.

    Returns:
        Path: A ROS Path message containing the waypoints.
    """
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rospy.Time.now()

    for seq, wp in enumerate(waypoints):
        x, y, yaw = wp
        pose_stamped = create_pose_stamped(x, y, yaw, path_msg.header.frame_id, seq, rospy.Time.now())
        path_msg.poses.append(pose_stamped)

    return path_msg


def create_pose_stamped(x: float, y: float, yaw: float, frame_id: str, seq: int, stamp: rospy.Time)->PoseStamped:
    """
    Creates a ROS PoseStamped message given position and orientation.

    Args:
        x (float): The x-coordinate of the position.
        y (float): The y-coordinate of the position.
        yaw (float): The orientation (yaw) in radians.
        frame_id (str): The frame of reference for the pose.
        seq (int): Sequence number of the pose.
        stamp (rospy.Time): Timestamp for the pose.

    Returns:
        PoseStamped: A ROS PoseStamped message containing the position and orientation.
    """
    quaternion = quaternion_from_euler(0, 0, yaw)
    pose_stamped = PoseStamped()
    pose_stamped.header.seq = seq
    pose_stamped.header.stamp = stamp
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.orientation.x = quaternion[0]
    pose_stamped.pose.orientation.y = quaternion[1]
    pose_stamped.pose.orientation.z = quaternion[2]
    pose_stamped.pose.orientation.w = quaternion[3]
    return pose_stamped

class GoalGenerator:
    """
    This class is responsible for generating and publishing navigation goals and paths using a pilot model.
    It listens to synchronized image and object detection messages, and processes them to generate a navigation
    goal in a specific coordinate frame.
    """

    def __init__(self):
        """
        Initializes the GoalGenerator class and sets up ROS nodes, subscribers, publishers, 
        and the pilot model for path planning and goal generation.
        """
        rospy.init_node('pilot_goal_generation_publisher', anonymous=True)
        
        # Initialize parameters
        params = self.load_parameters()

        # Subscribe to the image and object detection topics
        self.image_sub = message_filters.Subscriber(params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(params["obj_det_topic"], ObjectsStamped)

        # Set up publishers
        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.seq = 1

        # Initialize TF listener for frame transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Load the pilot model
        data_cfg, datasets_cfg, policy_model_cfg, encoder_model_cfg, device = get_inference_config(params["model_name"])
        self.image_size = data_cfg.image_size

        # Initialize queues for context and target data
        self.latest_image = None
        self.latest_obj_det = None
        self.context_queue = []
        self.target_context_queue = []
        self.context_size = data_cfg.context_size + 1
        self.target_context_size = self.context_size if data_cfg.target_context else 1
        self.max_depth = datasets_cfg.max_depth
        
        # Configure device for PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

        # Initialize pilot model
        self.model = PilotPlanner(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                encoder_model_cfg=encoder_model_cfg,
                                robot=params["robot"],
                                wpt_i=params["wpt_i"],
                                frame_rate=params["frame_rate"])

        self.transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")
        
        self.model.load(params["model_name"])
        self.model.to(device=device)

        # Set up the processing rate
        self.rate = rospy.Rate(params["frame_rate"])
        self.goal_to_target = np.array([1.0, 0.0])

        # Synchronize incoming messages
        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=[self.image_sub, self.obj_det_sub],
            queue_size=10,
            slop=0.1)
        self.ats.registerCallback(self.topics_sync_callback)

        self.odom_frame = params["odom_frame"]
        self.base_frame = params["base_frame"]
        
        
        self.new_data_available = False  # Flag to indicate new data is ready to be processed
        
        rospy.on_shutdown(self.shutdownhook)
        rospy.loginfo("GoalGenerator initialized successfully.")


    def load_parameters(self):
        """
        Load configuration parameters from the pilot_goal_generation_params.yaml file,
        and log them in a structured manner.
        """
        node_name = rospy.get_name()
        
        # Load parameters from the ROS parameter server
        params = {
            "robot": rospy.get_param(node_name + "/robot",  default="turtlebot"),
            "model_name": rospy.get_param(node_name + "/model/model_name", default="pilot-turtle-static-follower_2024-05-02_12-38-32"),
            "frame_rate": rospy.get_param(node_name + "/model/frame_rate", default=6),
            "wpt_i": rospy.get_param(node_name + "/model/wpt_i", default=2),
            "image_topic": rospy.get_param(node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "obj_det_topic": rospy.get_param(node_name + "/topics/obj_det_topic", default="/obj_detect_publisher_node/object"),
            "odom_frame": rospy.get_param(node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(node_name + "/frames/base_frame", default="base_link"),
        }
        
        # Log the loaded parameters in a structured way
        rospy.loginfo(f"******* {node_name} Parameters *******")
        rospy.loginfo("* Robot: " + params["robot"])
        
        rospy.loginfo("* Model:")
        rospy.loginfo("  * model_name: " + params["model_name"])
        rospy.loginfo("  * frame_rate: " + str(params["frame_rate"]))
        rospy.loginfo("  * wpt_i: " + str(params["wpt_i"]))
        
        rospy.loginfo("* Topics:")
        rospy.loginfo("  * image_topic: " + params["image_topic"])
        rospy.loginfo("  * obj_det_topic: " + params["obj_det_topic"])
        
        rospy.loginfo("* Frames:")
        rospy.loginfo("* odom_frame: " + params["odom_frame"])
        
        rospy.loginfo("**************************")
        
        return params
    
    def shutdownhook(self):
        """
        Function to be called on shutdown. Performs cleanup actions before exiting.
        """
        rospy.logwarn("Shutting down GoalGenerator.")
        # Additional cleanup actions can be added here.

    def topics_sync_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped):
        """
        Stores the latest received messages.
        """
        # rospy.loginfo("Image and object detection data synchronized.")
        self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
        self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] if obj_det_msg.objects else None
        self.new_data_available = True

    def maintain_queues(self):
        """
        Maintains the context and target queues by appending the latest image and object detection data.
        Ensures that the queues do not exceed their predefined sizes.

        The `context_queue` holds recent image data to provide context for navigation. The `target_context_queue`
        maintains positional data about detected objects as targets. If either queue exceeds its maximum size, 
        older data is discarded.

        The function also resets the `new_data_available` flag to indicate that the most recent data has been processed.

        Returns:
            None
        """
        # rospy.loginfo("Maintaining queues")

        # Append the latest image and object detection data to the respective queues
        self.context_queue.append(self.latest_image)
        self.target_context_queue.append(self.latest_obj_det)

        # Ensure that the queues do not exceed their maximum sizes
        if len(self.context_queue) > self.context_size:
            self.context_queue.pop(0)  # Remove the oldest element from the context queue

        if len(self.target_context_queue) > self.target_context_size:
            self.target_context_queue.pop(0)  # Remove the oldest element from the target context queue

        # Reset the flag indicating that new data has been processed
        self.new_data_available = False

    def generate(self):
        """
        Continuously processes synchronized data to generate navigation waypoints using
        the pilot model. Publishes the calculated goal pose to a ROS topic.
        """
        seq = 0

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # Check if new data has is available since last processed
            if self.new_data_available:
                
                # Maintain queues
                self.maintain_queues()
                
                if (len(self.context_queue) >= self.context_size and
                    len(self.target_context_queue) >= self.target_context_size):

                        # Prepare tensors for context and target data
                        context_queue_tensor = transform_images(self.context_queue[-self.context_size:], self.transform)
                        target_context_queue = self.target_context_queue[-self.target_context_size:]
                        target_context_queue_tensor = from_numpy(np.array(target_context_queue)).reshape(-1)
                        goal_to_target_tensor = from_numpy(self.goal_to_target)

                        # Perform inference to predict the next waypoint
                        # t = tic()
                        waypoint = self.model(context_queue_tensor, target_context_queue_tensor, goal_to_target_tensor)
                        # dt_infer = toc(t)
                        # rospy.loginfo(f"Inference time: {dt_infer} seconds")

                        dx, dy, hx, hy = waypoint
                        yaw = clip_angle(np.arctan2(hy, hx))

                        # Create a PoseStamped message for the calculated goal
                        pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, seq, current_time)
                        seq += 1

                        # Transform the goal pose to the odom frame
                        try:
                            transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time.now(), rospy.Duration(0.2))
                            transformed_pose = do_transform_pose(pose_stamped, transform)
                        except Exception as e:
                            rospy.logwarn(f"Failed to transform pose: {str(e)}")
                            continue
                        
                        rospy.loginfo_throttle(3, f"Planner running. Goal genretaed ([dx,dy,yaw]): [{dx},{dy},{yaw}] ")
                        # Publish the transformed goal pose
                        self.goal_pub.publish(transformed_pose)
                        # dt_process = toc(t)
                        # rospy.loginfo(f"Total processing time: {dt_process} seconds")

            self.rate.sleep()

class GoalGeneratorNoCond:
    def __init__(self):
        rospy.init_node('pilot_goal_generation_publisher', anonymous=True)
        
        # Initialize parameters
        params = self.load_parameters()

        # Load the pilot model
        data_cfg, datasets_cfg, policy_model_cfg, encoder_model_cfg, device = get_inference_config(params["model_name"])
        self.image_size = data_cfg.image_size
        self.max_depth = datasets_cfg.max_depth

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber(params["image_topic"], Image, self.image_callback)

        # Set up publishers
        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.seq = 1

        # Initialize TF listener for frame transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)



        # Initialize queues for context and target data
        self.latest_image = None
        self.context_queue = []
        self.context_size = data_cfg.context_size + 1
        
        self.new_data_available = False
        
        # Configure device for PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

        # Initialize pilot model
        self.model = PilotPlanner(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                encoder_model_cfg=encoder_model_cfg,
                                robot=params["robot"],
                                wpt_i=params["wpt_i"],
                                frame_rate=params["frame_rate"])

        self.model.load(params["model_name"])
        self.model.to(device=device)

        self.transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")

        # Set up the processing rate
        self.frame_rate = params["frame_rate"]
        self.pub_rate = params["pub_rate"]
        self.rate = rospy.Rate(self.pub_rate)
        self.pub_counter = 0

        self.goal_to_target = np.array([1.0, 0.0])

        self.odom_frame = params["odom_frame"]
        self.base_frame = params["base_frame"]

        rospy.on_shutdown(self.shutdownhook)
        rospy.loginfo("GoalGenerator initialized successfully.")

    def load_parameters(self):
        node_name = rospy.get_name()
        
        # Load parameters from the ROS parameter server
        params = {
            "robot": rospy.get_param(node_name + "/robot",  default="turtlebot"),
            "model_name": rospy.get_param(node_name + "/model/model_name", default="pilot-turtle-static-follower_2024-05-02_12-38-32"),
            "frame_rate": rospy.get_param(node_name + "/model/frame_rate", default=6),
            "pub_rate": rospy.get_param(node_name + "/model/pub_rate", default=1),  # Added pub_rate parameter
            "wpt_i": rospy.get_param(node_name + "/model/wpt_i", default=2),
            "image_topic": rospy.get_param(node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "odom_frame": rospy.get_param(node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(node_name + "/frames/base_frame", default="base_link"),
        }
        
        # Log the loaded parameters in a structured way
        rospy.loginfo(f"******* {node_name} Parameters *******")
        rospy.loginfo("* Robot: " + params["robot"])
        
        rospy.loginfo("* Model:")
        rospy.loginfo("  * model_name: " + params["model_name"])
        rospy.loginfo("  * frame_rate: " + str(params["frame_rate"]))
        rospy.loginfo("  * pub_rate: " + str(params["pub_rate"]))
        rospy.loginfo("  * wpt_i: " + str(params["wpt_i"]))
        
        rospy.loginfo("* Topics:")
        rospy.loginfo("  * image_topic: " + params["image_topic"])
        
        rospy.loginfo("* Frames:")
        rospy.loginfo("* odom_frame: " + params["odom_frame"])
        
        rospy.loginfo("**************************")
        
        return params

    def shutdownhook(self):
        rospy.logwarn("Shutting down GoalGenerator.")
        # Additional cleanup actions can be added here.

    def image_callback(self, image_msg: Image):
        self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
        self.context_queue.append(self.latest_image)
        if len(self.context_queue) > self.context_size:
            self.context_queue.pop(0)
        self.new_data_available = True

    def generate(self):
        seq = 0

        while not rospy.is_shutdown():
            

            if self.new_data_available == True:
                
                if len(self.context_queue) >= self.context_size:
                    context_queue_tensor = transform_images(self.context_queue[-self.context_size:], transform=self.transform)
                    goal_to_target_tensor = from_numpy(self.goal_to_target)

                    # Perform inference to predict the next waypoint
                    t = tic()
                    waypoint = self.model(context_queue_tensor, curr_rel_pos_to_target=None, goal_rel_pos_to_target=goal_to_target_tensor)

                    dt_infer = toc(t)
                    rospy.loginfo(f"Inference time: {dt_infer} seconds")
                        
                    dx, dy, hx, hy = waypoint
                    yaw = clip_angle(np.arctan2(hy, hx))
                    current_time = rospy.Time.now()
                    pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, seq, current_time)
                    seq += 1

                    try:
                        transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time.now(), rospy.Duration(0.4))
                        transformed_pose = do_transform_pose(pose_stamped, transform)
                    except Exception as e:
                        rospy.logwarn(f"Failed to transform pose: {str(e)}")
                        continue
                        
                    rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx,dy,yaw]): [{dx},{dy},{yaw}] ")
                    self.goal_pub.publish(transformed_pose)

                self.new_data_available = False

            self.pub_counter += 1
            self.rate.sleep()
if __name__ == '__main__':

    # Start node
    goal_gen = GoalGeneratorNoCond()
    goal_gen.generate()