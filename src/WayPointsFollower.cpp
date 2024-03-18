#include "waypoints_follower_control/WaypointsFollowerControl.hpp"

// STD
#include <string>

namespace wfc {

WaypointsFollowerControl::WaypointsFollowerControl(ros::NodeHandle& nodeHandle)
    : nodeHandle_(nodeHandle),
    prev_time_(ros::Time::now())
{
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }
    subscriber_ = nodeHandle_.subscribe(subscriberTopic_, 1,
                                    &WaypointsFollowerControl::goalCallback, this);
    cmd_publisher_ = nodeHandle_.advertise<geometry_msgs::Twist>(cmdTopic_, 10);
  ROS_INFO("Successfully launched node.");
}

WaypointsFollowerControl::~WaypointsFollowerControl()
{
}

bool WaypointsFollowerControl::readParameters()
{
    if (!nodeHandle_.getParam("subscriber_topic", subscriberTopic_)) return false;
    if (!nodeHandle_.getParam("subscriber_topic", cmdTopic_)) return false;

    return true;
}

void WaypointsFollowerControl::goalCallback(const geometry_msgs::PoseStamped& msg)
{   

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = msg.pose.orientation;
    tf2::Matrix3x3(tf2::Quaternion(quat.x, quat.y, quat.z, quat.w))
        .getRPY(roll, pitch, yaw);
    
    goal_xyyaw_in_odom_ << msg.pose.position.x, msg.pose.position.y, yaw; 
    bool print_goal = true;
    if (print_goal) {
        ROS_INFO("Goal | X: % .2f | Y: % .2f | | Yaw: % .2f ",
                goal_xyyaw_in_odom_[0], goal_xyyaw_in_odom_[1], goal_xyyaw_in_odom_[2]);
    }

    controller_.setGoal(goal_xyyaw_in_odom_);

}

void WaypointsFollowerControl::odomCallback(const nav_msgs::Odometry& msg)
{   

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = msg.pose.pose.orientation;
    tf2::Matrix3x3(tf2::Quaternion(quat.x, quat.y, quat.z, quat.w))
        .getRPY(roll, pitch, yaw);
    
    curr_xyyaw_in_odom_ << msg.pose.pose.position.x, msg.pose.pose.position.y, yaw; 
    
    bool print_odom = true;
    if (print_odom) {
        ROS_INFO("Odometry | X: % .2f | Y: % .2f | | Yaw: % .2f ",
                curr_xyyaw_in_odom_[0], curr_xyyaw_in_odom_[1], curr_xyyaw_in_odom_[2]);
    }

    // Get the end time
    ros::Time curr_time = ros::Time::now();

    // Calculate the time difference
    ros::Duration time_difference = curr_time - prev_time_;
    double dt = time_difference.toSec();

    Eigen::Vector2d control_cmd;
    controller_.getControl(curr_xyyaw_in_odom_,dt,control_cmd);

    geometry_msgs::Twist cmd_msg;
    cmd_msg.linear.x = control_cmd[0];
    cmd_msg.linear.y = 0.0;
    cmd_msg.linear.z = 0.0;
    cmd_msg.angular.z = control_cmd[1];
    cmd_msg.angular.y = 0.0;
    cmd_msg.angular.x = 0.0;
    cmd_publisher_.publish(cmd_msg);

    bool print_cmd = true;
    if (print_cmd) {
        ROS_INFO("Command | X: % .2f | Y: % .2f | Z: % .2f | Roll: % .2f | "
                "Pitch: % .2f | Yaw: % .2f ",
                cmd_msg.linear.x, cmd_msg.linear.y, cmd_msg.linear.z,
                cmd_msg.angular.x, cmd_msg.angular.y, cmd_msg.angular.z);
    }
}

} /* namespace */