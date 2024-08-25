#include "waypoints_follower_control/WaypointsFollowerControl.hpp"
// #include "waypoints_follower_control/ParametersConfig.h"
#include "waypoints_follower_control/WFCConfig.h"
// STD
#include <string>

namespace wfc {

WaypointsFollowerControl::WaypointsFollowerControl(ros::NodeHandle& nodeHandle)
    : nodeHandle_(nodeHandle),
    prev_time_(ros::Time::now()),
    goal_xyyaw_in_odom_({0.,0.,0.}),
    prev_raw_control_cmd_({0.,0.}),
    prev_filtered_control_cmd_({0.,0.}),
    k_(1.0),  // You can adjust these coefficients
    e_(0.0),  // You can adjust these coefficients
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

    ctrl_timer_ = nodeHandle.createTimer(ros::Duration(1./ctrl_loop_freq_),
                                    &WaypointsFollowerControl::ControlTimerCallback, this);

    // reach_goal_publisher_ = nodeHandle_.advertise<geometry_msgs::Twist>(cmdTopic_, 10);
    // PID controller(lin_Kp_, lin_vel_max_, lin_vel_min_,
    //                     ang_Kp_, ang_vel_max_, ang_vel_min_,
    //                     rotate_dist_threshold_);
    controller_ = new PID(lin_Kp_, lin_Ki_, lin_Kd_, lin_vel_max_, lin_vel_min_,
                            ang_Kp_, ang_Ki_, ang_Kd_,ang_vel_max_, ang_vel_min_,
                            rotate_dist_threshold_);

    dynamic_reconfigure::Server<waypoints_follower_control::WFCConfig>::CallbackType f;
    f = boost::bind(&WaypointsFollowerControl::cfgCallback, this, _1, _2);
    server_.setCallback(f);

    ROS_INFO("Successfully launched node.");
}

WaypointsFollowerControl::~WaypointsFollowerControl()
{
}


void WaypointsFollowerControl::cfgCallback(waypoints_follower_control::WFCConfig &config, uint32_t level) {
        ROS_INFO("Reconfigure Request: rotate_dist_threshold = %f,\n linear_kp = %f, linear_ki = %f, linear_kd = %f,\n angular_kp = %f, angular_ki = %f, angular_kd = %f,\n smoothing_k = %f",
                        config.rotate_dist_threshold,
                        config.linear_kp, config.linear_ki, config.linear_kd,
                        config.angular_kp, config.angular_ki, config.angular_kd,
                        config.smoothing_k);

        rotate_dist_threshold_ = config.rotate_dist_threshold;

        lin_Kp_ = config.linear_kp;
        lin_Ki_ = config.linear_ki;
        lin_Kd_ = config.linear_kd;

        ang_Kp_ = config.angular_kp;
        ang_Ki_ = config.angular_ki;
        ang_Kd_ = config.angular_kd;

        k_ = config.smoothing_k;
}


bool WaypointsFollowerControl::readParameters()
{
    if (!nodeHandle_.getParam("topics/goal_topic", goalTopic_)) return false;
    if (!nodeHandle_.getParam("topics/cmd_topic", cmdTopic_)) return false;
    if (!nodeHandle_.getParam("topics/odom_topic", odomTopic_)) return false;
    
    if (!nodeHandle_.getParam("control/ctrl_loop_freq", ctrl_loop_freq_)) return false;
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

    // Load smoothing parameters
    if (!nodeHandle_.getParam("control/smoothing/k", k_)) return false;
    if (!nodeHandle_.getParam("control/smoothing/e", e_)) return false;

    ROS_INFO_STREAM("******* Parameters *******");
    ROS_INFO_STREAM("* Topics:");
    ROS_INFO_STREAM("  * goal_topic: " << goalTopic_);
    ROS_INFO_STREAM("  * cmd_topic: " << cmdTopic_);
    ROS_INFO_STREAM("  * odom_topic: " << odomTopic_);
    ROS_INFO_STREAM("* Control:");
    ROS_INFO_STREAM("  * Control loop frequency: " << ctrl_loop_freq_ << " [hz]");

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

    ROS_INFO_STREAM("  * Smoothing:");
    ROS_INFO_STREAM("      * k: " << k_);
    ROS_INFO_STREAM("      * e: " << e_);

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
        ROS_INFO_THROTTLE(1.,"Goal | X: % .2f | Y: % .2f | | Yaw: % .2f ",
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
        ROS_INFO_THROTTLE(1.,"Odometry | X: % .2f | Y: % .2f | | Yaw: % .2f ",
                curr_xyyaw_in_odom_[0], curr_xyyaw_in_odom_[1], curr_xyyaw_in_odom_[2]);
    }
}


void WaypointsFollowerControl::ControlTimerCallback(const ros::TimerEvent&)
{
    
    // Get the end time
    ros::Time curr_time = ros::Time::now();
    ROS_INFO_STREAM("      * kp: " << lin_Kp_);

    // Calculate the time difference
    ros::Duration time_difference = curr_time - prev_time_;
    double dt = time_difference.toSec();
    prev_time_ = curr_time;
    
    Eigen::Vector2d control_cmd;
    controller_->getControl(curr_xyyaw_in_odom_,dt,control_cmd);

    // TODO: smooth control_cmd[0] and control_cmd[0] with the prev raw and the prev_filtered
    // filtered_control_cmd[0] = k*control_cmd[0] + e*prev_raw_control_cmd[0] + (1-e-k)*prev_filtered_control_cmd

    // Smooth control_cmd with the previous values
    filtered_control_cmd_[0] = k_ * control_cmd[0] + e_ * prev_raw_control_cmd_[0] + (1 - e_ - k_) * prev_filtered_control_cmd_[0];
    filtered_control_cmd_[1] = k_ * control_cmd[1] + e_ * prev_raw_control_cmd_[1] + (1 - e_ - k_) * prev_filtered_control_cmd_[1];

    // Update the previous command values
    prev_raw_control_cmd_ = control_cmd;
    prev_filtered_control_cmd_ = filtered_control_cmd_;

    geometry_msgs::Twist cmd_msg;
    cmd_msg.linear.x = filtered_control_cmd_[0];
    cmd_msg.linear.y = 0.0;
    cmd_msg.linear.z = 0.0;
    cmd_msg.angular.z = filtered_control_cmd_[1];
    cmd_msg.angular.y = 0.0;
    cmd_msg.angular.x = 0.0;
    cmd_publisher_.publish(cmd_msg);

    bool print_cmd = true;
    if (print_cmd) {
        ROS_INFO_THROTTLE(1.,"Command | X: % .2f | Y: % .2f | Z: % .2f | Roll: % .2f | "
                "Pitch: % .2f | Yaw: % .2f ",
                cmd_msg.linear.x, cmd_msg.linear.y, cmd_msg.linear.z,
                cmd_msg.angular.x, cmd_msg.angular.y, cmd_msg.angular.z);
    }


}



} /* namespace */