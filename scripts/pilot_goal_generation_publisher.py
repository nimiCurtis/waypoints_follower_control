#!/usr/bin/env python3
from typing import Tuple, List, Deque
from collections import deque

import numpy as np
import torch

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer,BufferInterface, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped


import message_filters
from zed_interfaces.msg import ObjectsStamped
from nav_msgs.msg import Odometry

from pilot_deploy.inference import PilotAgent, get_inference_config

from pilot_utils.transforms import transform_images, ObservationTransform
from pilot_utils.deploy.deploy_utils import msg_to_pil
from pilot_utils.deploy.modules import MovingWindowFilter
from pilot_utils.utils import tic, toc, from_numpy, normalize_data, xy_to_d_cos_sin, clip_angle
from pilot_utils.data.data_utils import to_local_coords

def pos_yaw_from_odom(odom_msg:Odometry)->list:
    pos = [odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z]
    ori = [odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w]
    
    euler = euler_from_quaternion(ori)
    yaw = euler[2]
    
    return [pos[0],pos[1],yaw]


def create_path_msg(waypoints: List[Tuple], frame_id: str) -> Path:
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

def create_pose_stamped(x: float, y: float, yaw: float, frame_id: str, seq: int, stamp: rospy.Time) -> PoseStamped:
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

def do_transform_pose_stamped(pose_stamped, transform):
    return tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

class MyBuffer(BufferInterface):
    def __init__(self):
        super(MyBuffer, self).__init__()
        self.buffer = Buffer()  # Use the existing tf2_ros Buffer
        self.listener = TransformListener(self.buffer)  # Listen for transforms
        self.registration.add(PoseStamped, do_transform_pose_stamped)

    def set_transform(self, transform, authority):
        self.buffer.set_transform(transform, authority)

    def lookup_transform(self, target_frame, source_frame, time, timeout=rospy.Duration(0.0)):
        return self.buffer.lookup_transform(target_frame, source_frame, time, timeout)

    def lookup_transform_full(self, target_frame, target_time, source_frame, source_time, fixed_frame, timeout=rospy.Duration(0.0)):
        return self.buffer.lookup_transform_full(target_frame, target_time, source_frame, source_time, fixed_frame, timeout)

    def can_transform(self, target_frame, source_frame, time, timeout=rospy.Duration(0.0)):
        return self.buffer.can_transform(target_frame, source_frame, time, timeout)

    def can_transform_full(self, target_frame, target_time, source_frame, source_time, fixed_frame, timeout=rospy.Duration(0.0)):
        return self.buffer.can_transform_full(target_frame, target_time, source_frame, source_time, fixed_frame, timeout)
class BaseGoalGenerator:
    def __init__(self):
        """
        Initializes the BaseGoalGenerator class, setting up ROS node, parameters, model configuration, and publishers/subscribers.
        """
        rospy.init_node('pilot_goal_generation_publisher', anonymous=True)

        # Load parameters
        params = self.load_parameters()
        self.params = params

        # Get inference configuration
        data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_cfg, linear_encoder_cfg, device = get_inference_config(params["model_name"])
        self.image_size = data_cfg.image_size
        self.max_depth = datasets_cfg.max_depth

        current_time = rospy.Time.now()
        self.last_collect_time = current_time
        self.last_inference_time = current_time
        self.last_msg_time = current_time

        # ROS publishers
        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.seq = 1

        # TF buffer and listener
        self.tf_buffer = MyBuffer()

        self.context_size = data_cfg.context_size
        self.action_context_size = data_cfg.action_context_size + 1 if data_cfg.action_context_size > 0 else data_cfg.action_context_size
        self.target_context = data_cfg.target_context
        self.target_dim = data_cfg.target_dim
        
        # Context queues
        self.context_queue: Deque = deque(maxlen=self.context_size + 1)
        self.target_context_queue: Deque = deque(maxlen=self.context_size + 1 if self.target_context else 1) # modify
        self.action_context_queue: Deque = deque(maxlen=self.action_context_size)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

        # Model initialization
        self.wpt_i = params["wpt_i"]
        self.model = PilotAgent(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                vision_encoder_cfg=vision_encoder_cfg,
                                linear_encoder_cfg=linear_encoder_cfg,
                                robot=params["robot"],
                                wpt_i=params["wpt_i"],
                                frame_rate=params["frame_rate"])

        self.model.load(model_name=params["model_name"],model_version=params["model_version"])
        self.model.to(device=device)

        self.transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")

        self.frame_rate = params["frame_rate"]
        self.inference_rate = params["inference_rate"]
        self.inference_times = deque(maxlen=self.inference_rate)

        self.odom_frame = params["odom_frame"]
        self.base_frame = params["base_frame"]

        self.transformed_pose = None
        self.transformed_pose_smoothed = None 
        self.smooth_goal_pos_filter = MovingWindowFilter(window_size=5,data_dim=3)
        self.smooth_goal_ori_filter = MovingWindowFilter(window_size=3,data_dim=4)

        self.prev_filtered_action = [0,0,0]
        self.k = 0.9
        
        rospy.on_shutdown(self.shutdownhook)

    def load_parameters(self):
        """
        Loads ROS parameters for the node.

        Returns:
            dict: A dictionary containing the loaded parameters.
        """
        node_name = rospy.get_name()

        params = {
            "robot": rospy.get_param(node_name + "/robot", default="turtlebot"),
            "model_name": rospy.get_param(node_name + "/model/model_name", default="pilot_tracking-bsz128_c3_ac3_td2_2024-08-06_21-35-26"),
            "model_version": str(rospy.get_param(node_name + "/model/model_version", default="best_model")),
            "frame_rate": rospy.get_param(node_name + "/model/frame_rate", default=7),
            "inference_rate": rospy.get_param(node_name + "/model/inference_rate", default=5),  # Added inference_rate parameter
            "wpt_i": rospy.get_param(node_name + "/model/wpt_i", default=2),
            "image_topic": rospy.get_param(node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "obj_det_topic": rospy.get_param(node_name + "/topics/obj_det_topic", default="/obj_detect_publisher_node/object"),
            "odom_topic": rospy.get_param(node_name + "/topics/odom_topic", default="/zedm/zed_node/odom"),
            "odom_frame": rospy.get_param(node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(node_name + "/frames/base_frame", default="base_link"),
        }

        rospy.loginfo(f"******* {node_name} Parameters *******")
        rospy.loginfo("* Robot: " + params["robot"])
        rospy.loginfo("* Model:")
        rospy.loginfo("  * model_name: " + params["model_name"])
        rospy.loginfo("  * model_version: " + params["model_version"])
        rospy.loginfo("  * frame_rate: " + str(params["frame_rate"]))
        rospy.loginfo("  * inference_rate: " + str(params["inference_rate"]))
        rospy.loginfo("  * wpt_i: " + str(params["wpt_i"]))
        rospy.loginfo("* Topics:")
        rospy.loginfo("  * image_topic: " + params["image_topic"])
        rospy.loginfo("  * obj_det_topic: " + params["obj_det_topic"])
        rospy.loginfo("  * odom_topic: " + params["odom_topic"])
        
        rospy.loginfo("* Frames:")
        rospy.loginfo("* odom_frame: " + params["odom_frame"])
        rospy.loginfo("* base_frame: " + params["base_frame"])

        rospy.loginfo("**************************")

        return params

    def shutdownhook(self):
        """
        ROS shutdown hook for cleanup actions.
        """
        rospy.logwarn("Shutting down GoalGenerator.")
        # Additional cleanup actions can be added here.

    def topics_callback(self, *args):
        """
        Abstract method to be implemented by derived classes for handling topic callbacks.
        """
        raise NotImplementedError("Derived classes must implement this method.")
    
    def filter_pose(self, *args):
        """
        Abstract method to be implemented by derived classes
        """
        raise NotImplementedError("Derived classes must implement this method.")

class GoalGenerator(BaseGoalGenerator):
    def __init__(self):
        """
        Initializes the GoalGenerator class, setting up subscribers and synchronizers for image and object detection topics.
        """
        super().__init__()
        self.goal_to_target = np.array([1.0, 0.0])
        # Subscribers and synchronizer for image and object detection topics
        self.image_sub = message_filters.Subscriber(self.params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(self.params["obj_det_topic"], ObjectsStamped)
        self.odom_sub = message_filters.Subscriber(self.params["odom_topic"], Odometry)
        
        self.sync_topics_list = [self.image_sub]

        if self.target_context:
            self.sync_topics_list.append(self.obj_det_sub)
            
            
        if self.action_context_size>0:
            self.sync_topics_list.append(self.odom_sub)

        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=self.sync_topics_list,
            queue_size=20,
            slop=0.2)
        
        
        self.ats.registerCallback(self.topics_callback)

        rospy.loginfo("GoalGenerator initialized successfully.")

    def topics_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped, odom_msg: Odometry = None):
        """
        Callback function for synchronized image and object detection messages. Processes data and performs inference.

        Args:
            image_msg (Image): Image message from the subscribed topic.
            obj_det_msg (ObjectsStamped): Object detection message from the subscribed topic.
        """
        current_time = image_msg.header.stamp

        # Collect image data at the specified frame rate
        dt_collect = (current_time - self.last_collect_time).to_sec()
        if dt_collect >= 1.0 / self.frame_rate:
            self.last_collect_time = current_time
            self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
            self.context_queue.append(self.latest_image)

            self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] if obj_det_msg.objects else [0, 0]
            self.target_context_queue.append(self.latest_obj_det)
            
            if odom_msg is not None:
                self.latest_odom_pos = pos_yaw_from_odom(odom_msg=odom_msg)
                self.action_context_queue.append(self.latest_odom_pos)

        # Perform inference at the specified inference rate
        dt_inference = (current_time - self.last_inference_time).to_sec()
        if (len(self.context_queue) >= self.context_queue.maxlen) and (len(self.target_context_queue) >= self.target_context_queue.maxlen) and  (len(self.action_context_queue) >= self.action_context_queue.maxlen) and (dt_inference >= 1.0 / self.inference_rate):
            self.last_inference_time = current_time

            # Transform image data and prepare target context tensor
            transformed_context_queue = transform_images(list(self.context_queue), transform=self.transform)
            target_context_queue = np.array(self.target_context_queue)
            
            prev_actions = None
            # action_context_queue = np.array(list(self.action_context_queue))
            if odom_msg is not None:
                action_context_queue = np.array(self.action_context_queue)
                
                prev_positions = action_context_queue[:,:2]
                prev_yaw = action_context_queue[:,2]
                prev_waypoints = to_local_coords(prev_positions, prev_positions[0], prev_yaw[0])
                prev_yaw = prev_yaw[1:] - prev_yaw[0]  # yaw is relative to the initial yaw
                prev_actions = np.concatenate([prev_waypoints[1:], prev_yaw[:, None]], axis=-1)
                prev_actions = from_numpy(prev_actions)


            target_context_mask = np.sum(target_context_queue == np.zeros((2,)), axis=1) == 2
            np_curr_rel_pos = np.zeros((target_context_queue.shape[0], self.target_dim))
            if self.target_dim == 3:
                np_curr_rel_pos[~target_context_mask] = xy_to_d_cos_sin(target_context_queue[~target_context_mask])
                np_curr_rel_pos[~target_context_mask, 0] = normalize_data(data=np_curr_rel_pos[~target_context_mask, 0], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})
            elif self.target_dim == 2:
                np_curr_rel_pos[~target_context_mask] = normalize_data(data=target_context_queue[~target_context_mask], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})
            
            # mask = np.sum(target_context_queue == np.zeros((2,)), axis=1) == 2
            # np_curr_rel_pos_in_d_theta = np.zeros((target_context_queue.shape[0], 3))
            # np_curr_rel_pos_in_d_theta[~mask] = xy_to_d_cos_sin(target_context_queue[~mask])
            # np_curr_rel_pos_in_d_theta[~mask, 0] = normalize_data(data=np_curr_rel_pos_in_d_theta[~mask, 0], stats={'min': 0.1, 'max': self.max_depth / 1000})
            target_context_queue_tensor = from_numpy(np_curr_rel_pos)

            # Prepare goal condition tensor
            if self.target_dim == 3:
                    goal_rel_pos_to_target = xy_to_d_cos_sin(self.goal_to_target)
                    goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})
            elif self.target_dim == 2:
                goal_rel_pos_to_target = normalize_data(data=self.goal_to_target, stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})

            # goal_rel_pos_to_target = xy_to_d_cos_sin(self.goal_to_target)
            # goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min': 0.1, 'max': self.max_depth / 1000})
            goal_to_target_tensor = from_numpy(goal_rel_pos_to_target)

            # Perform inference to get waypoints
            t = tic()
            waypoints = self.model(transformed_context_queue,
                                target_context_queue_tensor,
                                goal_to_target_tensor,
                                prev_actions)
            dt_infer = toc(t)
            # rospy.loginfo(f"Inferencing time: {dt_infer:.4f} seconds.")
            self.inference_times.append(dt_infer)
            avg_inference_time = np.mean(self.inference_times)
            rospy.loginfo_throttle(10, f"Average inference time (last {len(self.inference_times)}): {avg_inference_time:.4f} seconds.")

            dx, dy, hx, hy = waypoints[self.wpt_i]
            
            dx = self.k*dx + (1-self.k)*self.prev_filtered_action[0]
            dy = self.k*dy + (1-self.k)*self.prev_filtered_action[1]

            yaw = clip_angle(self.k*clip_angle(np.arctan2(hy, hx)) + (1-self.k)*self.prev_filtered_action[2]) 

            # smooth
            self.prev_filtered_action = [dx,dy,yaw]

            # Create and transform pose
            pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, self.seq, current_time)
            self.seq += 1
            rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx, dy, yaw]): [{dx:.4f}, {dy:.4f}, {yaw:.4f}]")
            try:
                # Transform the pose to the odom frame
                self.transformed_pose = self.tf_buffer.transform(object_stamped=pose_stamped,
                                                                target_frame=self.odom_frame,
                                                                timeout=rospy.Duration(0.2),
                                                                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                self.transformed_pose = None  # Ensure the transformed_pose is not used if transformation fails

        # Publish the transformed pose
        if self.transformed_pose is not None:
            dt_pub = (current_time - self.last_msg_time).to_sec()
            
            self.transformed_pose_smoothed = self.filter_pose(self.transformed_pose)
            self.goal_pub.publish(self.transformed_pose_smoothed)
            # rospy.loginfo(f"Publishing goal after {dt_pub} seconds.")
            self.last_msg_time = current_time


    def filter_pose(self, pose_stamped: PoseStamped):
            """
            """
            raw_pos = np.array([pose_stamped.pose.position.x,
                                pose_stamped.pose.position.y,
                                pose_stamped.pose.position.z])
            
            raw_or = np.array([pose_stamped.pose.orientation.x,
                                pose_stamped.pose.orientation.y,
                                pose_stamped.pose.orientation.z,
                                pose_stamped.pose.orientation.w])
            
            
            smoothed_pos = self.smooth_goal_pos_filter.calculate_average(raw_pos)
            
            pose_stamped_filtered = PoseStamped()
            pose_stamped_filtered.header.seq = pose_stamped.header.seq
            pose_stamped_filtered.header.stamp = pose_stamped.header.stamp
            pose_stamped_filtered.header.frame_id = pose_stamped.header.frame_id
            pose_stamped_filtered.pose.position.x = smoothed_pos[0]
            pose_stamped_filtered.pose.position.y = smoothed_pos[1]
            pose_stamped_filtered.pose.orientation.x = raw_or[0]
            pose_stamped_filtered.pose.orientation.y = raw_or[1]
            pose_stamped_filtered.pose.orientation.z = raw_or[2]
            pose_stamped_filtered.pose.orientation.w = raw_or[3]
            
            return pose_stamped_filtered
        
        



# class GoalGeneratorNoCond(BaseGoalGenerator):
#     def __init__(self):
#         super().__init__()

#         self.image_sub = rospy.Subscriber(self.params["image_topic"], Image, self.topics_callback)

#         rospy.loginfo("GoalGeneratorNoCond initialized successfully.")
        
#     def topics_callback(self, image_msg: Image):
#         current_time = rospy.Time.now()
#         dt_collect = (current_time - self.last_collect_time).to_sec()

#         if dt_collect >= 1.0 / self.frame_rate:
#             self.last_collect_time = current_time
#             self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
#             self.context_queue.append(self.latest_image)
            
#             if len(self.context_queue) > self.context_size:
#                 self.context_queue.pop(0)

#         dt_pub = (current_time - self.last_inference_time).to_sec()
#         if len(self.context_queue) >= self.context_size and dt_pub >= 1.0 / self.inference_rate:
#             self.last_inference_time = current_time

#             trasformed_context_queue = transform_images(self.context_queue[-self.context_size:], transform=self.transform)

#             waypoints = self.model(trasformed_context_queue)
#             dx, dy, hx, hy = waypoints[self.wpt_i]
#             yaw = clip_angle(np.arctan2(hy, hx))
            
#             path = create_path_msg(waypoints=waypoints, frame_id=self.base_frame)
#             pose_stamped = create_pose_stamped(dx, dy, yaw, self.base_frame, self.seq, current_time)
#             self.seq += 1
            
#             try:
#                 transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time.now(), rospy.Duration(0.2))
#                 transformed_pose = do_transform_pose(pose_stamped, transform)
#                 # transformed_pose = self.tf_buffer.transform(object_stamped=pose_stamped,
#                 #                                             target_frame=self.odom_frame,
#                 #                                             timeout=rospy.Duration(0.2))

#             except Exception as e:
#                 rospy.logwarn(f"Failed to transform pose: {str(e)}")
#                 pass
            
#             self.goal_pub.publish(transformed_pose)
#             self.path_pub.publish(path)
#             rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx, dy, yaw]): [{dx}, {dy}, {yaw}]")

if __name__ == '__main__':
    # Start node
    goal_gen = GoalGenerator()
    rospy.spin()
