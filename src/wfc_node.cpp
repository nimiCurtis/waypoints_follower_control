#include <ros/ros.h>
#include "waypoints_follower_control/WaypointsFollowerControl.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "waypoints_follower_control");
  ros::NodeHandle nodeHandle("~");

  wfc::WaypointsFollowerControl wfc(nodeHandle);

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}