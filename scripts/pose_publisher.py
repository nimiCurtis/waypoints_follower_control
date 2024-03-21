#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from pilot_deploy.src.inference import InferenceDataset, InferenceModel
from pilot_deploy.src.utils import msg_to_pil, tic, toc
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

class GoalGenerator:
    def __init__(self):
        rospy.init_node('path_and_goal_publisher', anonymous=True)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("/zedm/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
    
        # Set the desired capture frequency (15 Hz)
        self.capture_rate = 20.
        self.capture_interval = rospy.Duration(1.0 / self.capture_rate)

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # Initialize the timer for image capture
        self.capture_timer = rospy.Timer(self.capture_interval, self.capture_image)

        # Initialize the latest image
        self.latest_image = None
        self.context_queue = []
        self.context_size = 5
        
        config_path = "/home/roblab20/dev/pilot/pilot_bc/pilot_deploy/config/config.yaml"
        self.model =InferenceModel(config_path=config_path)
        self.rate = rospy.Rate(10)

    # Image callback function
    def image_callback(self, msg):
        # Store the received image
        self.latest_image = msg_to_pil(msg)
        print("collect image")

    # Timer callback function for image capture
    def capture_image(self, event):
        if self.latest_image is not None:
            obs_image = self.latest_image
            # Process the captured image (e.g., save to file, perform analysis)
            # Here you can implement the desired functionality for processing the image
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_image)
                
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_image)
            
            
            print("add image")
            # Clear the image buffer
            self.latest_image = None
            
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
        
            if(len(self.context_queue)>self.context_size):
                t = tic()
                waypoint = self.model.predict(self.context_queue)
                dt_infer = toc(t)
                print(f"inference time: {dt_infer}[sec]")
                
                dx, dy, hx, hy = waypoint

                yaw = self.clip_angle(np.arctan2(hy,hx))
                
                pose_stamped = self.create_pose_stamped(dx, dy, yaw, 'base_footprint', seq, rospy.Time.now())
                seq+=1

                try:
                    transform = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time())
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