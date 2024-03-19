#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler

def create_path_msg(waypoints, frame_id):
    path_msg = Path()
    path_msg.header.frame_id = frame_id  # Assuming the frame_id is "map"
    path_msg.header.stamp = rospy.Time.now()

    for seq, wp in enumerate(waypoints):
        x, y, yaw = wp
        pose_stamped = create_pose_stamped(x, y, yaw, path_msg.header.frame_id, seq, rospy.Time.now())
        path_msg.poses.append(pose_stamped)

    return path_msg


def create_pose_stamped(x, y, yaw, frame_id, seq, stamp):
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

if __name__ == '__main__':
    rospy.init_node('path_and_goal_publisher', anonymous=True)
    path_pub = rospy.Publisher('/poses_path', Path, queue_size=10)
    goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

    # Generate the waypoints
    num_waypoints = 60
    radius = 1.0
    circle_center = [1.0, 1.0]
    waypoints_circle = [[circle_center[0] + radius * np.cos(2 * np.pi * i / num_waypoints),
                         circle_center[1] + radius * np.sin(2 * np.pi * i / num_waypoints),
                         2 * np.pi * i / num_waypoints + np.pi / 2] for i in range(num_waypoints)]

    # Create the Path message and populate it with PoseStamped messages for each waypoint
    path_msg = create_path_msg(waypoints=waypoints_circle,frame_id="odom")

    i = 0
    rate = rospy.Rate(3)  # Set rate to 1 Hz
    for pose_stamped in path_msg.poses:
        if rospy.is_shutdown():
            break
        path_pub.publish(path_msg)
        goal_pub.publish(pose_stamped)  # Publish each waypoint as the goal
        rate.sleep()  # Wait for 1 second before publishing the next waypoint

    rospy.signal_shutdown("end!")