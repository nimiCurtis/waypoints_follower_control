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

from pilot_deploy.inference import PilotAgent, get_inference_config
from pilot_utils.transforms import transform_images, ObservationTransform
from pilot_utils.deploy.deploy_utils import msg_to_pil
from pilot_utils.utils import tic, toc, from_numpy, normalize_data, xy_to_d_cos_sin

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
        x, y, hx, hy = wp
        yaw = clip_angle(np.arctan2(hy, hx)) 
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

class BaseGoalGenerator:
    def __init__(self):
        rospy.init_node('pilot_goal_generation_publisher', anonymous=True)
        
        params = self.load_parameters()
        self.params = params
        
        data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_cfg, linear_encoder_cfg, device = get_inference_config(params["model_name"])
        self.image_size = data_cfg.image_size
        self.max_depth = datasets_cfg.max_depth

        self.last_collect_time = rospy.Time.now()
        self.last_pub_time = rospy.Time.now()

        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.seq = 1

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.context_queue = []
        self.context_size = data_cfg.context_size + 1
        self.target_context_size = self.context_size if data_cfg.target_context else 1        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

        self.wpt_i = params["wpt_i"]
        self.model = PilotAgent(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                vision_encoder_cfg=vision_encoder_cfg,
                                linear_encoder_cfg=linear_encoder_cfg,
                                robot=params["robot"],
                                wpt_i=params["wpt_i"],
                                frame_rate=params["frame_rate"])

        self.model.load(params["model_name"])
        self.model.to(device=device)

        self.transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")

        self.frame_rate = params["frame_rate"]
        self.pub_rate = params["pub_rate"]
        self.rate = rospy.Rate(self.pub_rate)

        self.odom_frame = params["odom_frame"]
        self.base_frame = params["base_frame"]

        rospy.on_shutdown(self.shutdownhook)

    def load_parameters(self):
        node_name = rospy.get_name()
        
        params = {
            "robot": rospy.get_param(node_name + "/robot",  default="turtlebot"),
            "model_name": rospy.get_param(node_name + "/model/model_name", default="pilot-turtle-static-follower_2024-05-02_12-38-32"),
            "frame_rate": rospy.get_param(node_name + "/model/frame_rate", default=6),
            "pub_rate": rospy.get_param(node_name + "/model/pub_rate", default=1),  # Added pub_rate parameter
            "wpt_i": rospy.get_param(node_name + "/model/wpt_i", default=2),
            "image_topic": rospy.get_param(node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "obj_det_topic": rospy.get_param(node_name + "/topics/obj_det_topic", default="/obj_detect_publisher_node/object"),
            "odom_frame": rospy.get_param(node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(node_name + "/frames/base_frame", default="base_link"),
        }
        
        rospy.loginfo(f"******* {node_name} Parameters *******")
        rospy.loginfo("* Robot: " + params["robot"])
        rospy.loginfo("* Model:")
        rospy.loginfo("  * model_name: " + params["model_name"])
        rospy.loginfo("  * frame_rate: " + str(params["frame_rate"]))
        rospy.loginfo("  * pub_rate: " + str(params["pub_rate"]))
        rospy.loginfo("  * wpt_i: " + str(params["wpt_i"]))
        rospy.loginfo("* Topics:")
        rospy.loginfo("  * image_topic: " + params["image_topic"])
        rospy.loginfo("  * obj_det_topic: " + params["obj_det_topic"])
        rospy.loginfo("* Frames:")
        rospy.loginfo("* odom_frame: " + params["odom_frame"])
        rospy.loginfo("**************************")
        
        return params

    def shutdownhook(self):
        rospy.logwarn("Shutting down GoalGenerator.")
        # Additional cleanup actions can be added here.

    def topics_callback(self, *args):
        raise NotImplementedError("Derived classes must implement this method.")

class GoalGenerator(BaseGoalGenerator):
    def __init__(self):
        super().__init__()
        self.target_context_queue = []
        self.goal_to_target = np.array([1.0, 0.0])

        self.image_sub = message_filters.Subscriber(self.params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(self.params["obj_det_topic"], ObjectsStamped)
        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=[self.image_sub, self.obj_det_sub],
            queue_size=10,
            slop=0.1)
        self.ats.registerCallback(self.topics_callback)
        
        rospy.loginfo("GoalGenerator initialized successfully.")

    def topics_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped):
        current_time = rospy.Time.now()
        dt_collect = (current_time - self.last_collect_time).to_sec()

        if dt_collect >= 1.0 / self.frame_rate:
            # rospy.loginfo(f"Image collected after {dt_collect} seconds.")

            self.last_collect_time = current_time
            self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
            self.context_queue.append(self.latest_image)

            
            # obj detection
            
            self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] if obj_det_msg.objects else [0,0]
            
            self.target_context_queue.append(self.latest_obj_det)
            
            if len(self.context_queue) > self.context_size:
                self.context_queue.pop(0)
            
            if len(self.target_context_queue) > self.target_context_size:
                self.target_context_queue.pop(0)

        dt_pub = (current_time - self.last_pub_time).to_sec()
        if len(self.context_queue) >= self.context_size and len(self.target_context_queue) >= self.target_context_size and dt_pub >= 1.0 / self.pub_rate:
            self.last_pub_time = current_time

            trasformed_context_queue = transform_images(self.context_queue[-self.context_size:], transform=self.transform)
            
            # Obj det 
            target_context_queue = self.target_context_queue[-self.target_context_size:]
            # To numpy
            target_context_queue = np.array(target_context_queue)

            mask = np.sum(target_context_queue==np.zeros((2,)),axis=1) == 2
            np_curr_rel_pos_in_d_theta = np.zeros((target_context_queue.shape[0],3))
            np_curr_rel_pos_in_d_theta[~mask] = xy_to_d_cos_sin(target_context_queue[~mask])
            np_curr_rel_pos_in_d_theta[~mask,0] = normalize_data(data=np_curr_rel_pos_in_d_theta[~mask,0], stats={'min':0.1,'max':self.max_depth/1000}) # max_depth in mm -> meters
            target_context_queue_tensor = from_numpy(np_curr_rel_pos_in_d_theta)
            
            # Goal condition
            goal_rel_pos_to_target = xy_to_d_cos_sin(self.goal_to_target)
            goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min':0.1,'max':self.max_depth/1000})
            goal_to_target_tensor = from_numpy(goal_rel_pos_to_target)

            t = tic()
            waypoints = self.model(trasformed_context_queue, target_context_queue_tensor, goal_to_target_tensor)
            dt_infer = toc(t)
            rospy.loginfo(f"Inferencing time: {dt_infer} seconds.")
            
            dx, dy, hx, hy = waypoints[self.wpt_i]
            yaw = clip_angle(np.arctan2(hy, hx))
            
            path = create_path_msg(waypoints=waypoints, frame_id=self.base_frame)
            pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, self.seq, current_time)
            self.seq += 1
            
            try:
                            # rospy.loginfo(f"Publishing goal after {dt_pub} seconds.")
                transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time.now(), rospy.Duration(0.2))
                transformed_pose = do_transform_pose(pose_stamped, transform)


                rospy.loginfo(f"Publishing goal after {dt_pub} seconds.")

                self.goal_pub.publish(transformed_pose)
                self.path_pub.publish(path)
                rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx, dy, yaw]): [{dx}, {dy}, {yaw}]")
            
            except Exception as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                
            
            

class GoalGeneratorNoCond(BaseGoalGenerator):
    def __init__(self):
        super().__init__()

        self.image_sub = rospy.Subscriber(self.params["image_topic"], Image, self.topics_callback)

        rospy.loginfo("GoalGeneratorNoCond initialized successfully.")
        
    def topics_callback(self, image_msg: Image):
        current_time = rospy.Time.now()
        dt_collect = (current_time - self.last_collect_time).to_sec()

        if dt_collect >= 1.0 / self.frame_rate:
            self.last_collect_time = current_time
            self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
            self.context_queue.append(self.latest_image)
            
            if len(self.context_queue) > self.context_size:
                self.context_queue.pop(0)

        dt_pub = (current_time - self.last_pub_time).to_sec()
        if len(self.context_queue) >= self.context_size and dt_pub >= 1.0 / self.pub_rate:
            self.last_pub_time = current_time

            trasformed_context_queue = transform_images(self.context_queue[-self.context_size:], transform=self.transform)

            waypoints = self.model(trasformed_context_queue)
            dx, dy, hx, hy = waypoints[self.wpt_i]
            yaw = clip_angle(np.arctan2(hy, hx))
            
            path = create_path_msg(waypoints=waypoints, frame_id=self.base_frame)
            pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, self.seq, current_time)
            self.seq += 1
            
            try:
                transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time.now(), rospy.Duration(0.2))
                transformed_pose = do_transform_pose(pose_stamped, transform)
            except Exception as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                pass
            
            self.goal_pub.publish(transformed_pose)
            self.path_pub.publish(path)
            rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx, dy, yaw]): [{dx}, {dy}, {yaw}]")

if __name__ == '__main__':
    # Start node
    goal_gen = GoalGenerator()
    rospy.spin()
