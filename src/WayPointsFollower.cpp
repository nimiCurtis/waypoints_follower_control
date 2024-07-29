#include "waypoints_follower_control/WaypointsFollowerControl.hpp"

// STD
#include <string>

namespace wfc {

WaypointsFollowerControl::WaypointsFollowerControl(ros::NodeHandle& nodeHandle)
    : nodeHandle_(nodeHandle),
    prev_time_(ros::Time::now()),
    goal_xyyaw_in_odom_({0.,0.,0.}),
    controller_(nullptr)
{
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }
    goal_subscriber_ = nodeHandle_.subscribe(goalTopic_, 1,
                                    &WaypointsFollowerControl::goalCallback, this);
    
    odom_subscriber_ = nodeHandle_.subscribe(odomTopic_, 1,
                                    &WaypointsFollowerControl::odomCallback, this);
    cmd_publisher_ = nodeHandle_.advertise<geometry_msgs::Twist>(cmdTopic_, 10);
    // reach_goal_publisher_ = nodeHandle_.advertise<geometry_msgs::Twist>(cmdTopic_, 10);
    // PID controller(lin_Kp_, lin_vel_max_, lin_vel_min_,
    //                     ang_Kp_, ang_vel_max_, ang_vel_min_,
    //                     rotate_dist_threshold_);
    controller_ = new PID(lin_Kp_, lin_Ki_, lin_Kd_, lin_vel_max_, lin_vel_min_,
                            ang_Kp_, ang_Ki_, ang_Kd_,ang_vel_max_, ang_vel_min_,
                            rotate_dist_threshold_);

    ROS_INFO("Successfully launched node.");
}

WaypointsFollowerControl::~WaypointsFollowerControl()
{
}

bool WaypointsFollowerControl::readParameters()
{
    if (!nodeHandle_.getParam("topics/goal_topic", goalTopic_)) return false;
    if (!nodeHandle_.getParam("topics/cmd_topic", cmdTopic_)) return false;
    if (!nodeHandle_.getParam("topics/odom_topic", odomTopic_)) return false;

    if (!nodeHandle_.getParam("control/rotate_dist_threshold", rotate_dist_threshold_)) return false;

    if (!nodeHandle_.getParam("control/angular/kp", ang_Kp_)) return false;
    if (!nodeHandle_.getParam("control/angular/ki", ang_Ki_)) return false; // Load angular Ki
    if (!nodeHandle_.getParam("control/angular/kd", ang_Kd_)) return false; // Load angular Kd
    if (!nodeHandle_.getParam("control/angular/max_vel", ang_vel_max_)) return false;
    if (!nodeHandle_.getParam("control/angular/min_vel", ang_vel_min_)) return false;

    if (!nodeHandle_.getParam("control/linear/kp", lin_Kp_)) return false;
    if (!nodeHandle_.getParam("control/linear/ki", lin_Ki_)) return false; // Load linear Ki
    if (!nodeHandle_.getParam("control/linear/kd", lin_Kd_)) return false; // Load linear Kd
    if (!nodeHandle_.getParam("control/linear/max_vel", lin_vel_max_)) return false;
    if (!nodeHandle_.getParam("control/linear/min_vel", lin_vel_min_)) return false;

    ROS_INFO_STREAM("******* Parameters *******");
    ROS_INFO_STREAM("* Topics:");
    ROS_INFO_STREAM("  * goal_topic: " << goalTopic_);
    ROS_INFO_STREAM("  * cmd_topic: " << cmdTopic_);
    ROS_INFO_STREAM("  * odom_topic: " << odomTopic_);
    ROS_INFO_STREAM("* Control:");
    ROS_INFO_STREAM("  * Rotate commands thresh: " << rotate_dist_threshold_);

    ROS_INFO_STREAM("  * Linear vel:");
    ROS_INFO_STREAM("      * max_vel: " << lin_vel_max_ << " | min_vel: " << lin_vel_min_);
    ROS_INFO_STREAM("      * kp: " << lin_Kp_);
    ROS_INFO_STREAM("      * ki: " << lin_Ki_);
    ROS_INFO_STREAM("      * kd: " << lin_Kd_);
    
    ROS_INFO_STREAM("  * Angular vel:");
    ROS_INFO_STREAM("      * max_vel: " << ang_vel_max_ << " | min_vel: " << ang_vel_min_);
    ROS_INFO_STREAM("      * kp: " << ang_Kp_);
    ROS_INFO_STREAM("      * ki: " << ang_Ki_);
    ROS_INFO_STREAM("      * kd: " << ang_Kd_);

    ROS_INFO_STREAM("**************************");

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
        ROS_INFO_THROTTLE(3.,"Goal | X: % .2f | Y: % .2f | | Yaw: % .2f ",
                goal_xyyaw_in_odom_[0], goal_xyyaw_in_odom_[1], goal_xyyaw_in_odom_[2]);
    }

    controller_->setGoal(goal_xyyaw_in_odom_);

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
        ROS_INFO_THROTTLE(3.,"Odometry | X: % .2f | Y: % .2f | | Yaw: % .2f ",
                curr_xyyaw_in_odom_[0], curr_xyyaw_in_odom_[1], curr_xyyaw_in_odom_[2]);
    }

    // Get the end time
    ros::Time curr_time = ros::Time::now();

    // Calculate the time difference
    ros::Duration time_difference = curr_time - prev_time_;
    double dt = time_difference.toSec();
    prev_time_ = curr_time;
    
    Eigen::Vector2d control_cmd;
    controller_->getControl(curr_xyyaw_in_odom_,dt,control_cmd);

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
        ROS_INFO_THROTTLE(3.,"Command | X: % .2f | Y: % .2f | Z: % .2f | Roll: % .2f | "
                "Pitch: % .2f | Yaw: % .2f ",
                cmd_msg.linear.x, cmd_msg.linear.y, cmd_msg.linear.z,
                cmd_msg.angular.x, cmd_msg.angular.y, cmd_msg.angular.z);
    }

}

} /* namespace */