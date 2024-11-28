#!/usr/bin/env python3
from typing import Tuple, List, Deque
from collections import deque

import numpy as np
import torch

import rospy
from geometry_msgs.msg import PoseStamped, Pose 
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer,BufferInterface, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

import message_filters
from zed_interfaces.msg import ObjectsStamped
from nav_msgs.msg import Odometry

from pilot_deploy.inference import PilotAgent, get_inference_config

from pilot_utils.transforms import transform_images, ObservationTransform
from pilot_utils.deploy.deploy_utils import msg_to_pil
from pilot_utils.deploy.modules import MovingWindowFilter, GoalPositionEstimator, RealtimeTraj, SubgoalsGen
from pilot_utils.utils import tic, toc, from_numpy, normalize_data, xy_to_d_cos_sin, clip_angles
from pilot_utils.data.data_utils import to_local_coords


from waypoints_follower_control.cfg import ParametersConfig
from dynamic_reconfigure.server import Server


GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"



class ROSTimers:
    def __init__(self, *timers):
        """
        Initialize the ROSTimers class with tuples of (name, rate).
        Each tuple will create a timer and a queue buffer of size equal to the rate.
        """
        self.timers = {}
        self.global_time = None  # To store the global current time
        for name, rate in timers:
            self.timers[name] = {
                "rate": rate,
                "last_tick": rospy.get_time(),
                "queue": deque(maxlen=rate)
            }

    # def tick(self):
    #     """
    #     Store the current ROS time globally.
    #     """
    #     self.global_time = rospy.get_time()

    def tick(self, time):
        """
        Store the current ROS time globally.
        """
        self.global_time = time

    def tock(self, name, current_time):
        """
        Calculate the time difference in seconds since the last tock, update the queue, and log the average delta time.
        """
        if name not in self.timers:
            rospy.logwarn(f"Timer {name} does not exist.")
            return None

        timer = self.timers[name]
        
        # Calculate delta time
        dt = current_time - timer["last_tick"]
        
        return dt

    def event(self, name):
        """
        Check if the tock(name) is within the limit of 1/<rate> for the given timer.
        """
        if name not in self.timers:
            rospy.logwarn(f"Timer {name} does not exist.")
            return False

        timer = self.timers[name]
        
        if self.global_time is None:
            rospy.logwarn("Global tick has not been recorded yet.")
            return None
        
        current_time = self.global_time

        dt = self.tock(name, current_time)
        # rospy.loginfo(f"[Timer: {name}] Dt: {dt:.6f} >= {dt:.6f}.")
        
        if dt is None:
            rospy.logwarn(f"Event check failed for timer {name}.")
            return False
        
        rate = timer["rate"]
        limit = 1.0 / rate

        if dt >= (limit - 0.033):
            # rospy.loginfo(f"[Timer: {name}] Event up: {dt:.6f} >= {limit:.6f}.")
            
            timer["last_tick"] = current_time
            
            timer["queue"].append(dt)

            # Calculate the average delta time
            # avg_dt = sum(timer["queue"]) / len(timer["queue"])
            avg_dt = np.mean(timer["queue"])
            # Log the average delta time every 3 seconds
            rospy.loginfo_throttle(
                2, f"[Timer: {name}] Avg rate: {1 / avg_dt:.6f} hz | Window size: {len(timer['queue'])}"
            )
            return True
        else:
            # rospy.logwarn(f"[Timer: {name}] Event exceeded limit: {dt:.6f} > {limit:.6f}.")
            return False

    def set_timer_rate(self,name, rate):
        timer = self.timers[name]
        
        rospy.loginfo(
                f"[Timer: {name}] Rate modified from: {timer['rate']} hz to {rate}"
            )
        
        timer["rate"] = rate
        timer['queue'] = deque(timer['queue'], maxlen=rate)


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

def create_path_msg(waypoints:zip, waypoints_frame, path_frame_id, seq, transform, time) -> Path:
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
        pose_stamped = do_transform_pose_stamped(pose_stamped=pose_stamped,transform=transform)
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
        time = rospy.Time.now()
        self.current_time = time
        self.last_collect_time = time
        self.last_inference_time = time

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
        self.target_context_enable = data_cfg.target_context_enable
        
        # add goal condition
        self.target_dim = data_cfg.target_dim

        # Frames
        self.odom_frame = self.params["odom_frame"]
        self.base_frame = self.params["base_frame"]
        
        # Setup inference timing
        
        
        self.frame_rate = self.params["frame_rate"]
        self.inference_rate = self.params["inference_rate"]
        

        # Not in use
        # self.pub_rate = self.params["pub_rate"]
        

        self.ros_timers = ROSTimers(("collect_data", self.frame_rate),
                                    ("inference", self.inference_rate))
        
        self.ros_timers.tick(rospy.get_time())
        
        self.srv = Server(ParametersConfig, self.cfg_callback)
        
        self.inference_times = deque(maxlen=self.inference_rate)
        
        
        self.context_queue = deque(maxlen=self.context_size + 1)
        self.target_context_queue = deque(maxlen=self.context_size + 1)
        self.action_context_queue = deque(maxlen=data_cfg.action_context_size + 1)
        
        rospy.loginfo_once(f"Memory container size: {self.frame_rate*2}")
        self.vision_memory_queue = deque(maxlen=max(self.frame_rate*2,1))
        self.vision_memory_msgs = deque(maxlen=max(self.frame_rate*2,1))
        self.linear_memory_queue = deque(maxlen=max(self.frame_rate*2,1))
        self.mem_time_delta = 0
        
        # Filter and goal settings
        self.goal_to_target = np.array([1.0, 0.0])
        self.observed_target = False
        self.latest_observed_obj_det = None
        
        self.smooth_goal_filter = MovingWindowFilter(window_size=self.params["sensor_moving_window_size"], data_dim=3)

        # Subgoal generator
        self.use_subgoal = self.params["use_subgoal"]

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
            "robot": rospy.get_param(self.node_name + "/robot", default="go2"),
            "model_name": rospy.get_param(self.node_name + "/model/model_name", default="pidiff_bsz256_c2_ac2_gcTrue_gcp0.75_ah16_ph32_tceTrue_ntmaxmin_2024-11-24_17-17-18"),
            "model_version": str(rospy.get_param(self.node_name + "/model/model_version", default="best_model")),
            "frame_rate": rospy.get_param(self.node_name + "/model/frame_rate", default=7),
            "pub_rate": rospy.get_param(self.node_name + "/model/pub_rate", default=10),
            "inference_rate": rospy.get_param(self.node_name + "/model/inference_rate", default=3),
            "wpt_i": rospy.get_param(self.node_name + "/model/wpt_i", default=2),
            "use_subgoal": rospy.get_param(self.node_name + "/model/use_subgoal", default=True),

            "image_topic": rospy.get_param(self.node_name + "/topics/image_topic", default="/zedm/zed_node/depth/depth_registered"),
            "obj_det_topic": rospy.get_param(self.node_name + "/topics/obj_det_topic", default="/obj_detect_publisher_node/object"),
            "odom_topic": rospy.get_param(self.node_name + "/topics/odom_topic", default="/zedm/zed_node/odom"),
            "odom_frame": rospy.get_param(self.node_name + "/frames/odom_frame", default="odom"),
            "base_frame": rospy.get_param(self.node_name + "/frames/base_frame", default="base_link"),
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
        rospy.loginfo("  * use_subgoal: " + str(params["use_subgoal"]))

        
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

    def _is_target_observed(self,obj_det_msg: ObjectsStamped)->bool:
        """
        Checks if a target is observed in the object detection message.

        Args:
            obj_det_msg (ObjectsStamped): Object detection message.

        Returns:
            bool: True if a target is observed, False otherwise.
        """
        return bool(obj_det_msg.objects)
    
    
    def cfg_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request:
                        frame_rate = {frame_rate}, 
                        wpt_i = {wpt_i}, 
                        smoothen_time = {smoothen_time}""".format(**config))

        # Update the corresponding variables in your class
        self.frame_rate = config.frame_rate
        self.ros_timers.set_timer_rate("collect_data", self.frame_rate)
        self.wpt_i = min(config.wpt_i, self.model.action_horizon-1)
        self.smoothen_time = config.smoothen_time

        return config

class GoalGenerator(BaseGoalGenerator):
    def __init__(self):
        """
        Initializes the GoalGenerator class, setting up subscribers and synchronizers for image and object detection topics.
        """
        super().__init__()
        
        self.threshold = 0.4
        # Subscribers and synchronizer for image and object detection topics
        self.image_sub = message_filters.Subscriber(self.params["image_topic"], Image)
        self.obj_det_sub = message_filters.Subscriber(self.params["obj_det_topic"], ObjectsStamped)
        self.odom_sub = message_filters.Subscriber(self.params["odom_topic"], Odometry)
        self.goal_pub_sensor = rospy.Publisher('/goal_pose_model', PoseStamped, queue_size=10)
        # Publisher for the subgoal marker
        self.subgoal_marker_pub = rospy.Publisher('/subgoal_marker', Marker, queue_size=10)
        self.memory_image_pub = rospy.Publisher('/memory', Image, queue_size=10)
        
        self.goal_reached_pub = rospy.Publisher('/goal_reach', Bool, queue_size=10)
        self.goal_reached = Bool(False)

        self.model_prediction_timer = rospy.Timer(rospy.Duration(1/self.inference_rate),
                                            self.model_prediction_callback)

        # Initialize the service client without waiting
        self.recording_service_client = rospy.ServiceProxy('/zion/zed_recording_node/record', SetBool)
        
        # Track if recording is active and if service is available
        self.goal_zone = 0.8
        self.recording_active = True
        self.recording_service_available = False
        self.goal_reached_run_timer = False
        # Timer to periodically check for service availability
        self.service_check_timer = rospy.Timer(rospy.Duration(5.0), self.check_recording_service)
        
        
        self.sync_topics_list = [self.image_sub]

        # if self.target_context_enable:
        self.sync_topics_list.append(self.obj_det_sub)

        self.use_action_context = False
        
        ## Add action history info
        if self.action_context_size>0:
            self.use_action_context = True
            self.sync_topics_list.append(self.odom_sub)

        # Initialize RealtimeTraj for managing and updating the trajectory
        self.realtime_traj = RealtimeTraj()
        self.start_time = rospy.Time.now()
        self.last_service_call_time = rospy.Time.now()
        self.last_goal_reached = rospy.Time.now()
        
        self.latest_image_msg = Image()
        self.subgoal_gen = SubgoalsGen(threshold=1.5)
        self.subgoal_to_target = None
        
        
        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=self.sync_topics_list,
            queue_size=10,
            slop=0.1)
        
        
        self.ats.registerCallback(self.topics_callback)


        rospy.loginfo("GoalGenerator initialized successfully.")

    def check_recording_service(self, event):
        """
        Timer callback to periodically check if the recording service is available.
        """
        try:
            rospy.wait_for_service('/zion/zed_recording_node/record', timeout=1.0)
            if not self.recording_service_available:
                rospy.loginfo("Recording service is now available.")
            self.recording_service_available = True
        except rospy.ROSException:
            if self.recording_service_available:
                rospy.logwarn("Recording service is unavailable.")
            self.recording_service_available = False



    def topics_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped, odom_msg: Odometry = None):
        """
        Callback function for synchronized image and object detection messages. Processes data and performs inference.

        Args:
            image_msg (Image): Image message from the subscribed topic.
            obj_det_msg (ObjectsStamped): Object detection message from the subscribed topic.
        """
        
        # Reset current time
        self.current_time = image_msg.header.stamp
        self.ros_timers.tick(self.current_time.to_sec())
        

        # Check for target detection
        self.observed_target = self._is_target_observed(obj_det_msg)
        
        # Set dt's
        dt_collect = (self.current_time - self.last_collect_time).to_sec()
        dt_inference = (self.current_time - self.last_inference_time).to_sec()

        # rospy.loginfo(f"dt: {dt_collect:.5f} | rate: {(1/dt_collect):.5f}")
        
        # ROS msgs to variables
        # Odom
        self.latest_odom_pos = pos_yaw_from_odom(odom_msg=odom_msg)
        # Detection
        self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] if self.observed_target else [0, 0]
        # Depth image
        self.latest_image_msg = image_msg
        
        # Check for goal reaching at topics rate.
        self.goal_reached.data = False
        
        
        # When target is detected -> calculate the relative distance. Then apply some logic.
        if self.observed_target:
            
            self.latest_observed_obj_det = np.array((self.latest_obj_det))
            
            goal_reached = self.is_goal_reached(self.latest_observed_obj_det, self.goal_to_target)
            self.goal_reached.data = goal_reached

            # Calc relative pose
            d_cos_sin_target_in_robot_base = xy_to_d_cos_sin(np.array(self.latest_obj_det))
            d_cos_sin_target_in_robot_base_desired = xy_to_d_cos_sin(self.goal_to_target)

            self.dgoal = d_cos_sin_target_in_robot_base[0] - d_cos_sin_target_in_robot_base_desired[0]
            
            desired_goal_pos_in_robot_base = np.array([self.dgoal*d_cos_sin_target_in_robot_base[1],self.dgoal*d_cos_sin_target_in_robot_base[2]])
            
            desired_goal_yaw_in_robot_base = np.arctan2(d_cos_sin_target_in_robot_base[2],d_cos_sin_target_in_robot_base[1])

            x_d, y_d, yaw_d = np.array([desired_goal_pos_in_robot_base[0],
                                        desired_goal_pos_in_robot_base[1],
                                        desired_goal_yaw_in_robot_base])
            
            
            desired_goal_quat_in_robot_base = quaternion_from_euler(0,0,yaw_d)

            # stamp_time = rospy.Time.now()
            desired_pose_stamped = create_pose_stamped([x_d, y_d], desired_goal_quat_in_robot_base, self.base_frame, self.seq, self.current_time)
            
            if not(self.goal_reached.data):
                
                if self.in_goal_zone_condition():
                    #### check conditions for zone goal

                    # Transform the desired pose to the odom frame
                    rospy.loginfo_throttle(1, f"Robot is in goal zone")
                    
                    try:
                        # Calc transform from base_frame to odom_frame
                        ros_transform = self.tf_buffer.lookup_transform(target_frame=self.odom_frame,
                                                                                    source_frame=self.base_frame,
                                                                                    time=self.current_time,
                                                                                    timeout=rospy.Duration(0.1))
                        
                        desired_pose_stamped_in_odom: PoseStamped = do_transform_pose_stamped(pose_stamped=desired_pose_stamped,
                                                                    transform=ros_transform)

                    except (LookupException, ConnectivityException, ExtrapolationException) as e:
                        rospy.logwarn(f"Failed to transform pose: {str(e)}")
                        desired_pose_stamped_in_odom = None  # Ensure the transformed_pose is not used if transformation fails

                    if desired_pose_stamped_in_odom is not None:
                            desired_pose_stamped_in_odom.header.seq = self.seq
                            self.seq+=1
                            self.goal_pub_sensor.publish(desired_pose_stamped_in_odom)

            else:
                
                ## Goal reached, means also observed target -> Control yaw!
                rospy.loginfo_throttle(1,f"{GREEN_COLOR}Goal reached!{RESET_COLOR}")
            
                pose_in_base = PoseStamped()
                
                yaw = clip_angles(np.arctan2(self.goal_to_target[1], self.goal_to_target[0]))
                
                q = quaternion_from_euler(0,0,yaw)
                
                pose_in_base.pose.orientation.x = q[0]
                pose_in_base.pose.orientation.y = q[1]
                pose_in_base.pose.orientation.z = q[2]
                pose_in_base.pose.orientation.w = q[3]
                
                pose_in_base.header.frame_id = self.base_frame
                # stamp_time = rospy.Time.now()
                pose_in_base.header.stamp = self.current_time 

                # Create and publish the updated path
                try:
                    # Calc transform from base_frame to odom_frame
                    ros_transform = self.tf_buffer.lookup_transform(target_frame=self.odom_frame,
                                                                                source_frame=self.base_frame,
                                                                                time=self.current_time,
                                                                                timeout=rospy.Duration(0.1))
                    pose_in_odom: PoseStamped  = do_transform_pose_stamped(pose_stamped=pose_in_base,transform=ros_transform)
                
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    rospy.logwarn(f"Failed to transform pose: {str(e)}")
                    pose_in_odom = None  # Ensure the transformed_pose is not used if transformation fails

                # Publish the transformed pose
                if pose_in_odom is not None:
                    pose_in_odom.header.seq = self.seq
                    self.seq+=1
                    self.goal_pub_sensor.publish(pose_in_odom)
        else:
            self.dgoal = None

        # Collect data at specified rate
        if self.ros_timers.event("collect_data"):
        # if dt_collect >= ((1/self.frame_rate) ) :
            # rospy.loginfo("collect data")
            self.last_collect_time = self.current_time

            self.latest_image = msg_to_pil(self.latest_image_msg, max_depth=self.max_depth)
            
            # Enques
            self.context_queue.append(self.latest_image)
            self.target_context_queue.append(self.latest_obj_det)
            self.action_context_queue.append(self.latest_odom_pos)

            if self.observed_target:
                ## Enque vision and lin memory
                rospy.loginfo_throttle(1, "Enque memory")
                self.vision_memory_queue.append(self.latest_image)
                self.linear_memory_queue.append(self.latest_observed_obj_det)
                self.vision_memory_msgs.append(self.latest_image_msg)
                self.mem_time_delta = 0
            else:
                self.mem_time_delta+=1

        self.goal_reached_pub.publish(self.goal_reached)

        if self.goal_reached.data and not(self.goal_reached_run_timer):
            self.goal_reached_run_timer = True
            self.last_goal_reached = self.current_time

        # Stop recording if the goal is reached
        if self.recording_service_available \
            and self.goal_reached_run_timer \
            and (self.current_time - self.last_goal_reached).to_sec() > 1 \
            and (self.current_time - self.last_service_call_time).to_sec() > 15.:  # Only stop if recording is currently active
            try:
                # Create a SetBool request with data=False to stop recording
                request = SetBoolRequest()
                request.data = False
                response = self.recording_service_client(request)
                # Update the last successful call time
                self.last_service_call_time = self.current_time
                self.goal_reached_run_timer = False
                
                if response.success:
                    rospy.loginfo(f"{GREEN_COLOR}Goal reached! Stopped recording successfully.{RESET_COLOR}")
                else:
                    rospy.logwarn("Failed to stop recording.")

                # Set recording_active to False as recording is now stopped
                # self.recording_active = False
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                self.recording_service_available = False

    def in_goal_zone_condition(self):
        
        if self.dgoal is None: 
            return False
        else:
            return self.dgoal <= self.goal_zone

    def model_prediction_callback(self, event):
        
        self.ros_timers.event("inference")
        
        
        # Queues are full
        if (
            (len(self.context_queue) >= self.context_queue.maxlen) 
            and (len(self.target_context_queue) >= self.target_context_queue.maxlen) 
            and  (len(self.action_context_queue) >= self.action_context_queue.maxlen)
            and not self.in_goal_zone_condition()
            and not self.goal_reached.data
        ):
            rospy.loginfo_throttle(2, f"Model is generating predictions..")
            # Transform image data and prepare target context tensor
            transformed_context_queue = transform_images(list(self.context_queue), transform=self.transform)
            target_context_queue = np.array(self.target_context_queue)

            # Prepare goal condition tensor
            if self.use_subgoal and self.observed_target:
                
                self.subgoal_to_target, radius = self.subgoal_gen.sample_subgoal(self.latest_observed_obj_det,self.goal_to_target)
                goal_to_target = self.subgoal_to_target
                rospy.loginfo_throttle(3,f"Target detected -> Using SubgoalGen, Publish marker")
                rospy.loginfo_throttle(3,f"Current target position {self.latest_observed_obj_det} | Subgoal generated: {goal_to_target}")
                # Publish the subgoal marker
                
                self.publish_subgoal_marker(self.latest_observed_obj_det, self.base_frame, radius)
            
            elif self.use_subgoal and self.subgoal_to_target is not None:
                rospy.loginfo_throttle(1,f"Target not detected -> Using last subgoal!")
                goal_to_target = self.subgoal_to_target
            else:
                # TODO: refactore -> always use use_subgoal
                ## Not in use!
                goal_to_target = self.goal_to_target
            
            
            ## TODO: change this
            goal_rel_pos_to_target = normalize_data(data=goal_to_target, stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin")

            prev_actions = None

            ## Take prev actions
            if self.use_action_context:
                action_context_queue = np.array(self.action_context_queue)
                
                prev_positions = action_context_queue[:,:2]
                prev_yaw = action_context_queue[:,2]
                prev_waypoints = to_local_coords(prev_positions, prev_positions[0], prev_yaw[0])
                prev_yaw = prev_yaw[1:] - prev_yaw[0]  # yaw is relative to the initial yaw
                prev_actions = np.concatenate([prev_waypoints[1:], prev_yaw[:, None]], axis=-1)
                prev_actions = from_numpy(prev_actions)

            
            ## Check if target in context
            #       - take the indices where there is no target
            target_context_mask = np.sum(target_context_queue == np.zeros((2,)), axis=1) == 2
            #       - check if in all of theme there is no target
            no_target_in_context = np.all(target_context_mask)
            
            
            ## Memory logic
            use_mem = False
            time_delta = 0
            if no_target_in_context and len(self.vision_memory_queue)>0:
                ## No target info at context at all
                # and there is a memory stored (target was detected in past):
                img_mem = self.vision_memory_queue[0]
                lin_mem = self.linear_memory_queue[0]
                img_msg = self.vision_memory_msgs[0]
                use_mem = True
                time_delta = self.mem_time_delta
                rospy.loginfo(f"No target in context. Using memory! --> time delta is {time_delta} ---> goal is {goal_to_target}, Publish image msg")
                self.memory_image_pub.publish(img_msg)
            else:
                ## When starting and target at frame
                img_mem = self.context_queue[-1]
                lin_mem = target_context_queue[-1]
                img_msg = self.latest_image_msg

            use_mem = torch.as_tensor(use_mem).long()
            time_delta = torch.as_tensor(time_delta, dtype=torch.float32)
            
            transformed_vision_memory_img = transform_images([img_mem], transform=self.transform)
            normalized_lin_mem = normalize_data(data=lin_mem, stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin" )

            np_curr_rel_pos = np.zeros((target_context_queue.shape[0], self.target_dim))
            np_curr_rel_pos[~target_context_mask] = normalize_data(data=target_context_queue[~target_context_mask], stats={'min': -self.max_depth / 1000, 'max': self.max_depth / 1000}, norm_type="maxmin" )

            target_context_queue_tensor = from_numpy(np_curr_rel_pos)

            normalized_lin_mem_tensor = from_numpy(normalized_lin_mem)
            goal_to_target_tensor = from_numpy(goal_rel_pos_to_target)

            # Perform inference to get waypoints
            t = tic()
            current_time_rel = (rospy.Time.now() - self.start_time).to_sec()
            
            ## TODO: insert to model
            waypoints = self.model(
                                transformed_context_queue,
                                target_context_queue_tensor,
                                goal_to_target_tensor,
                                prev_actions,
                                transformed_vision_memory_img,
                                normalized_lin_mem_tensor,
                                use_mem,
                                time_delta
                                )
            dt_infer = toc(t)
            # rospy.loginfo(f"Inferencing time: {dt_infer:.4f} seconds.")
            self.inference_times.append(dt_infer)
            avg_inference_time = np.mean(self.inference_times)
            rospy.loginfo_throttle(10, f"Average inference time (last {len(self.inference_times)}): {avg_inference_time:.4f} seconds.")


            # Umi on legs
            # Extract translations and quaternions from waypoints
            translations = np.array([[wp[0], wp[1], 0.0] for wp in waypoints])  # Assuming z=0.0
            quaternions_xyzw = np.array([quaternion_from_euler(0, 0, np.arctan2(wp[3], wp[2])) for wp in waypoints])
            timestamps = np.array([current_time_rel + ((i) / self.frame_rate) for i in range(len(waypoints))])

            # Update the trajectory with the new predictions using RealtimeTraj
            self.realtime_traj.update(
                translations=translations,
                quaternions_xyzw=quaternions_xyzw,
                timestamps=timestamps,
                current_timestamp= current_time_rel + dt_infer,
                smoothen_time=self.smoothen_time  # Smooth transition over 1 second
            )
            
            # Retrieve the smoothed trajectory for publishing
            smoothed_translations, smoothed_quaternions = self.realtime_traj.interpolate_traj(timestamps)

            
            # Create and publish the updated path
            try:
                # Create and publish the updated path
                # current_time = rospy.Time.now()
                # Calc transform from base_frame to odom_frame
                ros_transform = self.tf_buffer.lookup_transform(target_frame=self.odom_frame,
                                                                            source_frame=self.base_frame,
                                                                            time=self.current_time,
                                                                            timeout=rospy.Duration(0.2))
                path = create_path_msg(zip(smoothed_translations, smoothed_quaternions, timestamps), waypoints_frame = self.base_frame,
                                        path_frame_id=self.odom_frame,
                                        seq=self.seq, transform=ros_transform,
                                        time=self.current_time)

                transformed_pose : PoseStamped = path.poses[self.wpt_i]
                
                
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                rospy.logwarn(f"Failed to transform pose: {str(e)}")
                transformed_pose = None  # Ensure the transformed_pose is not used if transformation fails
                path = None

                            # Publish the transformed pose
            if transformed_pose is not None:
                transformed_pose.header.seq = self.seq
                self.seq+=1
                self.goal_pub_sensor.publish(transformed_pose)
                self.path_pub.publish(path)


    def is_goal_reached(self,latest_rel_pose:np.ndarray, goal_rel_pose:np.ndarray)->bool:
        # Calculate the Euclidean distance between the latest_rel_pose and goal_rel_pose
        distance = np.linalg.norm(latest_rel_pose - goal_rel_pose)
        
        # Check if the distance is within the threshold
        return distance <= self.threshold
    
    
    def publish_subgoal_marker(self, subgoal_position, frame_id, radius):
        """
        Publishes the subgoal as a Marker message in the base_link frame.

        Args:
            subgoal_position (np.ndarray): The position of the subgoal in the base_link frame.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "subgoal_marker"
        marker.id = 0
        marker.type = Marker.CYLINDER  # Using CYLINDER to create a ring appearance
        marker.action = Marker.ADD

        # Set the position of the marker
        marker.pose.position.x = subgoal_position[0]
        marker.pose.position.y = subgoal_position[1]
        marker.pose.position.z = 0.0  # Adjust the z-position if needed

        # Optional: Set orientation of the marker if needed
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Define the scale of the marker for a ring-like effect
        marker.scale.x = 2*radius  # Outer diameter of the ring
        marker.scale.y = 2*radius  # Outer diameter of the ring
        marker.scale.z = 0.02  # Small height to make it look like a ring in the XY plane

        # Define the color of the marker
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.3  # Alpha (transparency), 1.0 is opaque

        # Publish the marker
        self.subgoal_marker_pub.publish(marker)


if __name__ == '__main__':
    # Start node
    # goal_gen = GoalGeneratorKalman()
    goal_gen = GoalGenerator()
    rospy.spin()




