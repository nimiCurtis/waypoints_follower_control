#!/usr/bin/env python3
from typing import Tuple, List, Deque
from collections import deque

import numpy as np
import torch

import rospy
from geometry_msgs.msg import PoseStamped, Pose 
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
from pilot_utils.deploy.modules import MovingWindowFilter, GoalPositionEstimator, RealtimeTraj
from pilot_utils.utils import tic, toc, from_numpy, normalize_data, xy_to_d_cos_sin, clip_angle
from pilot_utils.data.data_utils import to_local_coords


from waypoints_follower_control.cfg import ParametersConfig
from dynamic_reconfigure.server import Server

def pos_yaw_from_odom(odom_msg:Odometry)->list:
    """
    Extracts position and yaw from a Odometry message.

    Args:
        odom_msg (Odometry): A ROS Odometry message.

    Returns:
        list: A list containing the x, y position and yaw.
    """
    return pos_yaw_from_pose(odom_msg.pose.pose)

def pos_yaw_from_pose(pose_msg: Pose) -> list:
    """
    Extracts position and yaw from a Pose message.

    Args:
        pose_msg (Pose): A ROS Pose message.

    Returns:
        list: A list containing the x, y position and yaw.
    """
    pos = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    ori = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    yaw = euler_from_quaternion(ori)[2]
    return [pos[0], pos[1], yaw]

def do_transform_pose_stamped(pose_stamped, transform):
    return tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)


def create_pose_stamped(translation, quaternion, frame_id: str, seq: int, stamp: rospy.Time) -> PoseStamped:
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
    
    pose_stamped = PoseStamped()
    pose_stamped.header.seq = seq
    pose_stamped.header.stamp = stamp #rospy.Time(stamp)
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose.position.x = translation[0]
    pose_stamped.pose.position.y = translation[1]
    pose_stamped.pose.orientation.x = quaternion[0]
    pose_stamped.pose.orientation.y = quaternion[1]
    pose_stamped.pose.orientation.z = quaternion[2]
    pose_stamped.pose.orientation.w = quaternion[3]
    return pose_stamped

def create_path_msg(waypoints:zip, waypoints_frame, path_frame_id, seq) -> Path:
    """
    Creates a ROS Path message from a list of waypoints.

    Args:
        waypoints (list of tuple): List of waypoints, where each waypoint is a tuple (x, y, yaw).
        frame_id (str): The frame of reference for the path.

    Returns:
        Path: A ROS Path message containing the waypoints.
    """
    
    path_msg = Path()
    path_msg.header.seq = seq
    path_msg.header.frame_id = path_frame_id
    current_time  = rospy.Time.now()
    path_msg.header.stamp = current_time

    seq = 0 
    for translation, quaternion, timestamp in waypoints:
        pose_stamped = create_pose_stamped(translation, quaternion, waypoints_frame, seq, current_time)
        # pose_stamped = do_transform_pose_stamped(pose_stamped=pose_stamped,transform=transform)
        path_msg.poses.append(pose_stamped)
        seq+=1

    return path_msg


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
        self.node_name = rospy.get_name()

        # Load parameters
        self.params = self.load_parameters()

        # Get inference configuration
        data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_cfg, linear_encoder_cfg, device = get_inference_config(self.params["model_name"])
        self.image_size = data_cfg.image_size
        self.max_depth = datasets_cfg.max_depth

        # Timing attributes
        current_time = rospy.Time.now()
        self.last_collect_time = current_time
        self.last_inference_time = current_time
        self.last_msg_time = current_time

        # ROS publishers
        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        
        # Sequence counter for messages
        self.seq = 1

        # TF buffer and listener
        self.tf_buffer = MyBuffer()

        # Model initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        self.wpt_i = self.params["wpt_i"]
        self.model = PilotAgent(
            data_cfg=data_cfg,
            policy_model_cfg=policy_model_cfg,
            vision_encoder_cfg=vision_encoder_cfg,
            linear_encoder_cfg=linear_encoder_cfg,
            robot=self.params["robot"],
            wpt_i=self.params["wpt_i"],
            frame_rate=self.params["frame_rate"]
        )
        self.model.load(model_name=self.params["model_name"], model_version=self.params["model_version"])
        self.model.to(device=device)

        # Transform and context setup
        self.transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")
        self.context_size = data_cfg.context_size
        self.action_context_size = data_cfg.action_context_size
        self.target_context = data_cfg.target_context
        self.target_dim = data_cfg.target_dim

        self.context_queue = deque(maxlen=self.context_size + 1)
        self.target_context_queue = deque(maxlen=self.context_size + 1 if self.target_context else 1)
        self.action_context_queue = deque(maxlen=data_cfg.action_context_size + 1)

        # Filter and goal settings
        self.goal_to_target = np.array([1.0, 0.0])
        self.observed_target = False
        
        self.smooth_goal_filter = MovingWindowFilter(window_size=self.params["sensor_moving_window_size"], data_dim=3)
        self.smoothen_time = self.params["smoothen_time"]
        # Setup inference timing
        self.frame_rate = self.params["frame_rate"]
        self.pub_rate = self.params["pub_rate"]
        self.inference_rate = self.params["inference_rate"]
        self.inference_times = deque(maxlen=self.inference_rate)

        # Frames
        self.odom_frame = self.params["odom_frame"]
        self.base_frame = self.params["base_frame"]

        self.transformed_pose = PoseStamped()
        self.ros_transform = None
        self.path = Path()
        # self.smooth_goal_ori_filter = MovingWindowFilter(window_size=3,data_dim=1)
        rospy.on_shutdown(self.shutdownhook)
        

    def load_parameters(self):
        """
        Loads ROS parameters for the node.

        Returns:
            dict: A dictionary containing the loaded parameters.
        """
        

        params = {
            "robot": rospy.get_param(self.node_name + "/robot", default="turtlebot"),
            "model_name": rospy.get_param(self.node_name + "/model/model_name", default="pilot_bsz160_c6_ac3_gcp0.4_mdp0.0_ph162024-08-24_15-16-54"),
            "model_version": str(rospy.get_param(self.node_name + "/model/model_version", default="best_model")),
            "frame_rate": rospy.get_param(self.node_name + "/model/frame_rate", default=7),
            "pub_rate": rospy.get_param(self.node_name + "/model/pub_rate", default=10),
            "inference_rate": rospy.get_param(self.node_name + "/model/inference_rate", default=3),
            "wpt_i": rospy.get_param(self.node_name + "/model/wpt_i", default=2),
            "image_topic": rospy.get_param(self.node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "obj_det_topic": rospy.get_param(self.node_name + "/topics/obj_det_topic", default="/obj_detect_publisher_node/object"),
            "odom_topic": rospy.get_param(self.node_name + "/topics/odom_topic", default="/zedm/zed_node/odom"),
            "odom_frame": rospy.get_param(self.node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(self.node_name + "/frames/base_frame", default="base_footprint"),
            "sensor_moving_window_size": rospy.get_param(self.node_name + "/filter/sensor_moving_window_size", default=1),
            "smoothen_time": rospy.get_param(self.node_name + "/filter/smoothen_time", default=0.1),

        }

        rospy.loginfo(f"******* {self.node_name} Parameters *******")
        rospy.loginfo("* Robot: " + params["robot"])
        rospy.loginfo("* Model:")
        rospy.loginfo("  * model_name: " + params["model_name"])
        rospy.loginfo("  * model_version: " + params["model_version"])
        rospy.loginfo("  * frame_rate: " + str(params["frame_rate"]))
        rospy.loginfo("  * pub_rate: " + str(params["pub_rate"]))

        rospy.loginfo("  * inference_rate: " + str(params["inference_rate"]))
        rospy.loginfo("  * wpt_i: " + str(params["wpt_i"]))
        rospy.loginfo("* Topics:")
        rospy.loginfo("  * image_topic: " + params["image_topic"])
        rospy.loginfo("  * obj_det_topic: " + params["obj_det_topic"])
        rospy.loginfo("  * odom_topic: " + params["odom_topic"])
        
        rospy.loginfo("* Frames:")
        rospy.loginfo("  * odom_frame: " + params["odom_frame"])
        rospy.loginfo("  * base_frame: " + params["base_frame"])

        rospy.loginfo("* Filter:")
        rospy.loginfo("  * sensor_moving_window_size: " + str(params["sensor_moving_window_size"]))
        rospy.loginfo("  * smoothen_time: " + str(params["smoothen_time"]))

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

    def _is_target_observed(self,obj_det_msg: ObjectsStamped)->bool:
        """
        Checks if a target is observed in the object detection message.

        Args:
            obj_det_msg (ObjectsStamped): Object detection message.

        Returns:
            bool: True if a target is observed, False otherwise.
        """
        return bool(obj_det_msg.objects)


class GoalGenerator(BaseGoalGenerator):
    def __init__(self):
        """
        Initializes the GoalGenerator class, setting up subscribers and synchronizers for image and object detection topics.
        """
        super().__init__()
        # Subscribers and synchronizer for image and object detection topics
        self.image_sub = message_filters.Subscriber(self.params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(self.params["obj_det_topic"], ObjectsStamped)
        self.odom_sub = message_filters.Subscriber(self.params["odom_topic"], Odometry)
        self.goal_pub_sensor = rospy.Publisher('/goal_pose_model', PoseStamped, queue_size=10)
        self.predcition_timer = rospy.Timer(rospy.Duration(1/self.inference_rate),
                                            self.prediction_callback)
        
        # self.goal_pub_timer = rospy.Timer(rospy.Duration(1/self.pub_rate),
        #                                     self.goal_pub_callback)

        self.srv = Server(ParametersConfig, self.cfg_callback)
        
        
        self.sync_topics_list = [self.image_sub]

        if self.target_context:
            self.sync_topics_list.append(self.obj_det_sub)

        self.use_action_context = False
        if self.action_context_size>0:
            self.use_action_context = True
            self.sync_topics_list.append(self.odom_sub)

        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=self.sync_topics_list,
            queue_size=10,
            slop=0.1)
        
        
        self.ats.registerCallback(self.topics_callback)
        
        # Initialize RealtimeTraj for managing and updating the trajectory
        self.realtime_traj = RealtimeTraj()
        self.start_time = rospy.Time.now()
        
        

        rospy.loginfo("GoalGenerator initialized successfully.")


    def cfg_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request:
                        frame_rate = {frame_rate}, 
                        wpt_i = {wpt_i}, 
                        smoothen_time = {smoothen_time}""".format(**config))
        
        # Update the corresponding variables in your class
        self.frame_rate = config.frame_rate
        # self.pub_rate = config.pub_rate
        # self.inference_rate = config.inference_rate
        self.wpt_i = config.wpt_i
        self.smoothen_time = config.smoothen_time

        return config
    
    def prediction_callback(self,event):
        
        # Perform inference at the specified inference rate
        if (len(self.context_queue) >= self.context_queue.maxlen) and (len(self.target_context_queue) >= self.target_context_queue.maxlen) and  (len(self.action_context_queue) >= self.action_context_queue.maxlen):

            # Transform image data and prepare target context tensor
            transformed_context_queue = transform_images(list(self.context_queue), transform=self.transform)
            target_context_queue = np.array(self.target_context_queue)
            
            prev_actions = None

            if self.use_action_context:
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
                np_curr_rel_pos[~target_context_mask, 0] = normalize_data(data=np_curr_rel_pos[~target_context_mask, 0], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin")
            elif self.target_dim == 2:
                np_curr_rel_pos[~target_context_mask] = normalize_data(data=target_context_queue[~target_context_mask], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin" )


            target_context_queue_tensor = from_numpy(np_curr_rel_pos)

            # Prepare goal condition tensor
            if self.target_dim == 3:
                    goal_rel_pos_to_target = xy_to_d_cos_sin(self.goal_to_target)
                    goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin")
            elif self.target_dim == 2:
                goal_rel_pos_to_target = normalize_data(data=self.goal_to_target, stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin")

            goal_to_target_tensor = from_numpy(goal_rel_pos_to_target)

            # Perform inference to get waypoints
            t = tic()
            current_time = (rospy.Time.now() - self.start_time).to_sec()
            waypoints = self.model(transformed_context_queue,
                                target_context_queue_tensor,
                                goal_to_target_tensor,
                                prev_actions)
            dt_infer = toc(t)
            # rospy.loginfo(f"Inferencing time: {dt_infer:.4f} seconds.")
            self.inference_times.append(dt_infer)
            avg_inference_time = np.mean(self.inference_times)
            rospy.loginfo_throttle(10, f"Average inference time (last {len(self.inference_times)}): {avg_inference_time:.4f} seconds.")

            
            # Umi on legs
            # Extract translations and quaternions from waypoints
            translations = np.array([[wp[0], wp[1], 0.0] for wp in waypoints])  # Assuming z=0.0
            quaternions_xyzw = np.array([quaternion_from_euler(0, 0, np.arctan2(wp[3], wp[2])) for wp in waypoints])
            timestamps = np.array([current_time + ((i + 1) / self.frame_rate) for i in range(len(waypoints))])

            # Update the trajectory with the new predictions using RealtimeTraj
            self.realtime_traj.update(
                translations=translations,
                quaternions_xyzw=quaternions_xyzw,
                timestamps=timestamps,
                current_timestamp= current_time + dt_infer,
                smoothen_time=self.smoothen_time  # Smooth transition over 1 second
            )
            
            # Retrieve the smoothed trajectory for publishing
            smoothed_translations, smoothed_quaternions = self.realtime_traj.interpolate_traj(timestamps)

            try:
                # self.ros_transform = self.tf_buffer.lookup_transform(target_frame=self.odom_frame,
                #                                                 source_frame=self.base_frame,
                #                                                 time = rospy.Time(0),
                #                                                 timeout=rospy.Duration(0.2))
                # Create and publish the updated path
                self.path = create_path_msg(zip(smoothed_translations, smoothed_quaternions, timestamps), waypoints_frame = self.base_frame,
                                        path_frame_id=self.base_frame,
                                        seq=self.seq) #, transform=self.ros_transform)

                self.transformed_pose: PoseStamped = self.path.poses[self.wpt_i]
                self.transformed_pose.header.seq = self.seq
                
                self.seq+=1
                
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                self.transformed_pose = None  # Ensure the transformed_pose is not used if transformation fails
                self.path = None
                
            
            if self.transformed_pose is not None:
                self.seq+=1
                self.transformed_pose_smoothed = self.filter_pose(self.transformed_pose)
                self.goal_pub_sensor.publish(self.transformed_pose_smoothed)
                # self.goal_pub_sensor.publish(self.transformed_pose)
                self.path_pub.publish(self.path)

    # def goal_pub_callback(self,event):
    #     # Publish the transformed pose
    #     if self.transformed_pose is not None:
    #         self.seq+=1
    #         self.transformed_pose_smoothed = self.filter_pose(self.transformed_pose)
    #         self.goal_pub_sensor.publish(self.transformed_pose_smoothed)
    #         # self.goal_pub_sensor.publish(self.transformed_pose)
    #         self.path_pub.publish(self.path)

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

        smoothed_pos = self.smooth_goal_filter.calculate_average(raw_pos)
        
        pose_stamped_filtered = PoseStamped()
        pose_stamped_filtered.header.seq = pose_stamped.header.seq
        pose_stamped_filtered.header.stamp = rospy.Time.now()
        pose_stamped_filtered.header.frame_id = pose_stamped.header.frame_id
        pose_stamped_filtered.pose.position.x = smoothed_pos[0]
        pose_stamped_filtered.pose.position.y = smoothed_pos[1]
        pose_stamped_filtered.pose.orientation.x = raw_or[0]
        pose_stamped_filtered.pose.orientation.y = raw_or[1]
        pose_stamped_filtered.pose.orientation.z = raw_or[2]
        pose_stamped_filtered.pose.orientation.w = raw_or[3]
        
        return pose_stamped_filtered



## TODO: update the kalman filter
class GoalGeneratorKalman(BaseGoalGenerator):
    def __init__(self):
        """
        Initializes the GoalGenerator class, setting up subscribers and synchronizers for image and object detection topics.
        """
        super().__init__()
        
        params = self.load_kalman_parameters()
        
        # Subscribers and synchronizer for image and object detection topics
        self.image_sub = message_filters.Subscriber(self.params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(self.params["obj_det_topic"], ObjectsStamped)
        self.odom_sub = message_filters.Subscriber(self.params["odom_topic"], Odometry)
        
        
        self.goal_pub_filtered = rospy.Publisher('/goal_pose_filtered', PoseStamped, queue_size=10)
        self.goal_pub_sensor = rospy.Publisher('/goal_pose_model', PoseStamped, queue_size=10)
        self.goal_pub_prediction = rospy.Publisher('/goal_pose_analytic', PoseStamped, queue_size=10)
        
        self.sync_topics_list = [self.image_sub]

        if self.target_context:
            self.sync_topics_list.append(self.obj_det_sub)

        if self.action_context_size>0:
            self.sync_topics_list.append(self.odom_sub)

        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=self.sync_topics_list,
            queue_size=10,
            slop=0.1)

        ## kalman filtering 
        # Initialize the filter parameters
        self.k_B = params["kalman_k_B"]
        self.sensor_variance = np.array([float(v) for v in params["kalman_sensor_variance"]])
        self.moving_window_size = params["kalman_moving_window_size"]
        self.alpha = params["kalman_alpha"]
        self.error_threhsold = params['kalman_error_threshold']

        self.goal_estimator = GoalPositionEstimator(sensor_variance=self.sensor_variance,
                                                    moving_window_filter_size=self.moving_window_size,
                                                    k=self.k_B)

        self.estimated_goal = np.zeros(3)
        self.desired_pose_in_odom = np.zeros(3)
        ## Register callback
        self.ats.registerCallback(self.topics_callback)

        rospy.loginfo("GoalGenerator initialized successfully.")

    def load_kalman_parameters(self):
        
        params = {
            # New parameters from the 'filter' section
            
            "kalman_k_B": rospy.get_param(self.node_name + "/filter/kalman/k_B", default=0.1),
            "kalman_sensor_variance": rospy.get_param(self.node_name + "/filter/kalman/sensor_variance", default=[1e-2, 1e-2, 1e-2]),
            "kalman_moving_window_size": rospy.get_param(self.node_name + "/filter/kalman/moving_window_size", default=15),
            "kalman_alpha": rospy.get_param(self.node_name + "/filter/kalman/alpha", default=150),
            "kalman_error_threshold": rospy.get_param(self.node_name + "/filter/kalman/error_threshold", default=0.6),
        }

        # Print the newly added filter parameters
        rospy.loginfo("* Filter Parameters:")
        rospy.loginfo(f"  * kalman_k_B: {params['kalman_k_B']}")
        rospy.loginfo(f"  * kalman_sensor_variance: {params['kalman_sensor_variance']}")
        rospy.loginfo(f"  * kalman_moving_window_size: {params['kalman_moving_window_size']}")
        rospy.loginfo(f"  * kalman_alpha: {params['kalman_alpha']}")
        rospy.loginfo(f"  * kalman_error_threshold: {params['kalman_error_threshold']}")

        rospy.loginfo("**************************")
        
        return params

    def topics_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped, odom_msg: Odometry = None):
        """
        Callback function for synchronized image and object detection messages. Processes data and performs inference.

        Args:
            image_msg (Image): Image message from the subscribed topic.
            obj_det_msg (ObjectsStamped): Object detection message from the subscribed topic.
        """
        current_time = image_msg.header.stamp
        self.observed_target = self._is_target_observed(obj_det_msg)
        model_prediction_in_odom = None
        
        # Collect image data at the specified frame rate
        dt_collect = (current_time - self.last_collect_time).to_sec()
        
        self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] if self.observed_target else [0, 0]
        obj_det_variance = np.array(obj_det_msg.objects[0].position_covariance) if self.observed_target else np.zeros(6)

        if dt_collect >= 1.0 / self.frame_rate:
            self.last_collect_time = current_time
            self.latest_image = msg_to_pil(image_msg, max_depth=self.max_depth)
            self.context_queue.append(self.latest_image)

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
            
            target_context_queue_tensor = from_numpy(np_curr_rel_pos)

            # Prepare goal condition tensor
            if self.target_dim == 3:
                    goal_rel_pos_to_target = xy_to_d_cos_sin(self.goal_to_target)
                    goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})
            elif self.target_dim == 2:
                goal_rel_pos_to_target = normalize_data(data=self.goal_to_target, stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000})

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

            dx_m, dy_m, hx_m, hy_m = waypoints[self.wpt_i]
            yaw_m = clip_angle(np.arctan2(hy_m, hx_m))
            
            dx_m, dy_m, yaw_m = self.smooth_goal_filter.calculate_average(np.array([dx_m,dy_m,yaw_m]))
            yaw_m = clip_angle(yaw_m)

            model_pose_stamped = create_pose_stamped(dx_m, dy_m, yaw_m, self.base_frame, self.seq, current_time)
            
            # smooth
            
            self.prev_filtered_action = np.array([dx_m,dy_m,yaw_m])

            try:
                # Transform the pose to the odom frame
                self.transformed_pose: PoseStamped = self.tf_buffer.transform(object_stamped=model_pose_stamped,
                                                                target_frame=self.odom_frame,
                                                                timeout=rospy.Duration(0.2),
                                                                )
                model_prediction_in_odom = pos_yaw_from_pose(self.transformed_pose.pose)
                
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                self.transformed_pose = None  # Ensure the transformed_pose is not used if transformation fails

        # Calculate error of current relative target position from the desired relative target position
        
        d_cos_sin_target_in_robot_base = xy_to_d_cos_sin(np.array(self.latest_obj_det))
        d_cos_sin_target_in_robot_base_desired = xy_to_d_cos_sin(self.goal_to_target)
        
        dgoal = d_cos_sin_target_in_robot_base[0] - d_cos_sin_target_in_robot_base_desired[0]
        
        desired_goal_pos_in_robot_base = np.array([dgoal*d_cos_sin_target_in_robot_base[1],dgoal*d_cos_sin_target_in_robot_base[2]])
        desired_goal_yaw_in_robot_base = np.arctan2(d_cos_sin_target_in_robot_base[2],d_cos_sin_target_in_robot_base[1])

        desired_pose_stamped = create_pose_stamped(desired_goal_pos_in_robot_base[0], desired_goal_pos_in_robot_base[1], desired_goal_yaw_in_robot_base, self.base_frame, self.seq, current_time)

        try:
            # Transform the pose to the odom frame
            desired_pose_stamped_in_odom: PoseStamped = self.tf_buffer.transform(object_stamped=desired_pose_stamped,
                                                            target_frame=self.odom_frame,
                                                            timeout=rospy.Duration(0.2),
                                                            )

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                desired_pose_stamped_in_odom = None  # Ensure the transformed_pose is not used if transformation fails

        desired_pose_in_odom = np.array(pos_yaw_from_pose(pose_msg=desired_pose_stamped_in_odom.pose))

        if self.observed_target:
            e = desired_pose_in_odom - self.estimated_goal
            # e[2] = clip_angle(e[2])
            rospy.loginfo(f"e: [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]")

            current_pose_in_odom = np.array(pos_yaw_from_odom(odom_msg=odom_msg))

            e2s = np.abs(desired_pose_in_odom - current_pose_in_odom)
            # e2s[2] = clip_angle(e2s[2])
            rospy.loginfo(f"e2: [{e2s[0]:.4f}, {e2s[1]:.4f}, {e2s[2]:.4f}]")

            e2 = np.linalg.norm(e2s)
            rospy.loginfo(f"e2 norm: {e2:.4f}")

            alpha = self.alpha
            v = np.round(np.exp(-alpha * (e2 - self.error_threhsold )),4)
            prediction_mag = (1/(v + 1e-6))  # need to increase prediction covariance when e2 is big, decrease when small 
            correction_mag =  v # need to increase prediction covariance when e2 is small, decrease when big
            nprediction_mag = prediction_mag / (correction_mag + prediction_mag)
            ncorrection_mag = correction_mag / (correction_mag + prediction_mag)
        else:
            e = np.zeros(3)
            nprediction_mag = 1.
            ncorrection_mag = 1e-20

        rospy.loginfo(f"Prediction cov magnitude: {nprediction_mag}")
        rospy.loginfo(f"Correction cov magnitude: {ncorrection_mag}")

        self.goal_estimator.update(state_error=e,
                                sensor_prediction=model_prediction_in_odom,
                                prediction_variance=obj_det_variance,
                                prediction_mag=nprediction_mag,
                                correction_mag=ncorrection_mag)
        
        self.estimated_goal = self.goal_estimator.estimated_goal
        
        dx, dy, yaw = self.estimated_goal[0], self.estimated_goal[1], clip_angle(self.estimated_goal[2])
        
        # Create and transform pose
        pose_stamped = create_pose_stamped(dx, dy, yaw, self.odom_frame, self.seq, current_time)
        self.seq += 1
        rospy.loginfo(f"Planner running. Goal generated ([dx, dy, yaw]): [{dx:.4f}, {dy:.4f}, {yaw:.4f}]")

        # rospy.loginfo_throttle(1, f"Planner running. Goal generated ([dx, dy, yaw]): [{dx:.4f}, {dy:.4f}, {yaw:.4f}]")

        # Publish the transformed pose
        if pose_stamped is not None:
            dt_pub = (current_time - self.last_msg_time).to_sec()
            # self.transformed_pose_smoothed = self.filter_pose(self.transformed_pose)
            self.goal_pub_filtered.publish(pose_stamped)
            # rospy.loginfo(f"Publishing goal after {dt_pub} seconds.")
            self.last_msg_time = current_time

        if self.transformed_pose is not None:
            self.goal_pub_sensor.publish(self.transformed_pose)

        if desired_pose_stamped_in_odom is not None:
            self.goal_pub_prediction.publish(desired_pose_stamped_in_odom)

if __name__ == '__main__':
    # Start node
    # goal_gen = GoalGeneratorKalman()
    goal_gen = GoalGenerator()
    rospy.spin()
