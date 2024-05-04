#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from zed_interfaces.msg import ObjectsStamped
from tf.transformations import quaternion_from_euler
from pilot_deploy.inference import PilotPlanner, get_inference_config
from pilot_utils.transforms import transform_images
from pilot_utils.deploy.deploy_utils import msg_to_pil
from pilot_utils.utils import tic, toc, from_numpy
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import torch
import message_filters


class GoalGenerator:
    def __init__(self):
        rospy.init_node('path_and_goal_publisher', anonymous=True)

        # Subscribe to the image topic
        self.image_sub = message_filters.Subscriber("/zedm/zed_node/depth/depth_registered", Image)
        self.obj_det_sub = message_filters.Subscriber("/obj_detect_publisher_node/object", ObjectsStamped)

        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.seq = 1
        # Set the desired capture frequency (15 Hz)
        self.capture_rate = 6.
        self.capture_interval = rospy.Duration(1.0 / self.capture_rate)

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # Initialize the timer for image capture
        # self.capture_timer = rospy.Timer(self.capture_interval, self.capture_image)

        
        
        model_name = "pilot-turtle-static-follower_2024-05-01_23-28-38"
        data_cfg, _, policy_model_cfg, encoder_model_cfg, device = get_inference_config(model_name=model_name)
        robot = "turtlebot"
        self.image_size = data_cfg.image_size
        
        # Initialize the latest image
        self.latest_image = None
        self.latest_obj_det = None
        self.context_queue = []
        self.target_context_queue = []
        self.context_size = data_cfg.context_size + 1
        self.target_context_size = self.context_size if data_cfg.target_context  else 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
    
        self.model = PilotPlanner(data_cfg=data_cfg,
                            policy_model_cfg=policy_model_cfg,
                            encoder_model_cfg=encoder_model_cfg,
                            robot=robot)
        
        self.model.load(model_name=model_name)
        self.model.to(device=device)

        self.rate = rospy.Rate(6)
        
        self.goal_to_target = np.array([1.0 ,0.0])
        
        # Synchronize the incoming messages based on their timestamp
        self.ats = message_filters.ApproximateTimeSynchronizer(
            fs=[self.image_sub, self.obj_det_sub],
            queue_size=10,
            slop=0.1)
        self.ats.registerCallback(self.topics_sync_callback) 

    def topics_sync_callback(self, image_msg: Image, obj_det_msg: ObjectsStamped):
        
        self.latest_image = msg_to_pil(image_msg) 

        # Check if there is at least one detection
        if len(obj_det_msg.objects) > 0:
            self.latest_obj_det = list(obj_det_msg.objects[0].position)[:2] ### TODO: obj_det_to_numpy
            
            print("add image")
            self.context_queue.append(self.latest_image)
            self.target_context_queue.append(self.latest_obj_det)
            
            if len(self.context_queue) > self.context_size:
                self.context_queue.pop(0)

            if len(self.target_context_queue) > self.target_context_size:
                self.target_context_queue.pop(0)
        
            # if(len(self.context_queue)>=self.context_size and len(self.target_context_queue)>=self.target_context_size):
                    
            #     context_queue_tensor = transform_images(self.context_queue[-self.context_size:], image_size=self.image_size)

                
            #     self.target_context_queue = self.target_context_queue[-self.target_context_size:]
                
            #     if len(self.target_context_queue) == 1:
            #         self.target_context_queue = self.target_context_queue[-1]

            #     target_context_queue_tensor = from_numpy(np.array(self.target_context_queue))
            #     goal_to_target_tensor = from_numpy(self.goal_to_target)
                
            #     t = tic()
            #     waypoint = self.model(context_queue_tensor,target_context_queue_tensor, goal_to_target_tensor)
            #     dt_infer = toc(t)
            #     print(f"inference time: {dt_infer}[sec]")

            #     dx, dy, hx, hy = waypoint

            #     yaw = self.clip_angle(np.arctan2(hy,hx))
                
                
            #     pose_stamped = self.create_pose_stamped(dx, dy, yaw, 'base_footprint', self.seq, rospy.Time.now())
            #     self.seq+=1
            #     try:
            #         transform = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time())
            #         transformed_pose = do_transform_pose(pose_stamped, transform)
            #     except Exception as e:
            #         rospy.logwarn("Failed to transform pose: %s", str(e))
            #         print("hi")
            #     # Publish the transformed pose
            #     # self.goal_pub.publish(pose_stamped)

            #     self.goal_pub.publish(transformed_pose)
            #     dt_process = toc(t)
            #     print(f"process time: {dt_process}[sec]")


    def clip_angle(self,theta) -> float:
        """Clip angle to [-pi, pi]"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi

    def create_path_msg(self,waypoints, frame_id):
        path_msg = Path()
        path_msg.header.frame_id = frame_id  # Assuming the frame_id is "map"
        path_msg.header.stamp = rospy.Time.now()

        for seq, wp in enumerate(waypoints):
            x, y, yaw = wp
            pose_stamped = self.create_pose_stamped(x, y, yaw, path_msg.header.frame_id, seq, rospy.Time.now())
            path_msg.poses.append(pose_stamped)

        return path_msg

    
    def create_pose_stamped(self,x, y, yaw, frame_id, seq, stamp):
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
    
    def generate(self):

        seq=0
        while not rospy.is_shutdown():
            
            if(len(self.context_queue)>=self.context_size and len(self.target_context_queue)>=self.target_context_size):
                
                context_queue_tensor = transform_images(self.context_queue[-self.context_size:], image_size=self.image_size)

                
                target_context_queue = self.target_context_queue[-self.target_context_size:]
                
                if len(target_context_queue) == 1:
                    target_context_queue = target_context_queue[-1]

                target_context_queue_tensor = from_numpy(np.array(target_context_queue))
                goal_to_target_tensor = from_numpy(self.goal_to_target)
                
                t = tic()
                waypoint = self.model(context_queue_tensor,target_context_queue_tensor, goal_to_target_tensor)
                dt_infer = toc(t)
                print(f"inference time: {dt_infer}[sec]")

                dx, dy, hx, hy = waypoint

                yaw = self.clip_angle(np.arctan2(hy,hx))
                
                pose_stamped = self.create_pose_stamped(dx, dy, yaw, 'base_link', seq, rospy.Time.now())
                seq+=1

                try:
                    transform = self.tf_buffer.lookup_transform("odom", "base_link", rospy.Time())
                    transformed_pose = do_transform_pose(pose_stamped, transform)
                except Exception as e:
                    rospy.logwarn("Failed to transform pose: %s", str(e))
                    continue
                
                # Publish the transformed pose
                # self.goal_pub.publish(pose_stamped)

                self.goal_pub.publish(transformed_pose)
                dt_process = toc(t)
                print(f"process time: {dt_process}[sec]")

            self.rate.sleep()




if __name__ == '__main__':

    goal_gen = GoalGenerator()
    goal_gen.generate()