#include <ros/ros.h>
#include "waypoints_follower_control/WaypointsFollowerControl.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ros_package_template");
  ros::NodeHandle nodeHandle("~");

  wfc::WaypointsFollowerControl wfc(nodeHandle);

  ros::spin();
  return 0;
}