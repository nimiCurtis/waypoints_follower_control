#pragma once

#include "PID.hpp"

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Temperature.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace std;

namespace wfc {

/*!
 * Main class for the node to handle the ROS interfacing.
 */
class WaypointsFollowerControl
{
 public:
    /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  WaypointsFollowerControl(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
    virtual ~WaypointsFollowerControl();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
    bool readParameters();

    /*!
   * ROS topic callback method.
   * @param message the received message.
   */
    void goalCallback(const geometry_msgs::PoseStamped& msg);

    /*!
   * ROS topic callback method.
   * @param message the received message.
   */
    void odomCallback(const nav_msgs::Odometry& msg);



    void ControlTimerCallback(const ros::TimerEvent&);
  //   /*!
  //    * ROS service server callback.
  //    * @param request the request of the service.
  //    * @param response the provided response.
  //    * @return true if successful, false otherwise.
  //    */
  //   bool serviceCallback(std_srvs::Trigger::Request& request,
  //                        std_srvs::Trigger::Response& response);

  //! ROS node handle.
    ros::NodeHandle& nodeHandle_;


  //! ROS topic subscriber.
    ros::Subscriber goal_subscriber_;
    ros::Subscriber odom_subscriber_;

    ros::Publisher cmd_publisher_;

    ros::Timer ctrl_timer_;
  //! ROS topic name to subscribe to.
    std::string goalTopic_;
    std::string cmdTopic_;
    std::string odomTopic_;

    Eigen::Vector3d curr_xyyaw_in_odom_;
    Eigen::Vector3d goal_xyyaw_in_odom_;    
    
    //! Algorithm computation object.
    PID* controller_;

    ros::Time prev_time_;

    double ang_Kp_;
    double ang_Ki_;
    double ang_Kd_;
    
    double lin_Kp_;
    double lin_Ki_;
    double lin_Kd_;

    double lin_vel_max_ ;
    double lin_vel_min_ ;

    double ang_vel_max_ ;
    double ang_vel_min_ ;

    double rotate_dist_threshold_;
    double ctrl_loop_freq_;

    // Added variables for smoothing control commands
    Eigen::Vector2d prev_raw_control_cmd_;       // Previous raw control command
    Eigen::Vector2d prev_filtered_control_cmd_;  // Previous filtered control command
    Eigen::Vector2d filtered_control_cmd_;       // Current filtered control command
    double k_;                                   // Coefficient for current command influence
    double e_;                                   // Coefficient for previous raw command influence
};

} /* namespace */