#include "waypoints_follower_control/PID.hpp"

#include <utility>

using namespace std;


double normalize_angle(double yaw) {
    
    // Normalize the angle between -π and π
    if (yaw > M_PI) {
        yaw -= 2 * M_PI;
    }
    else if (yaw < -M_PI) {
        yaw += 2 * M_PI;
    }
    else{
        return yaw;
    }

    return normalize_angle(yaw);
}


namespace wfc {


PID::PID() {
  goal_xyyaw_in_odom_ = std::make_unique<Eigen::Vector3d>();
}

PID::~PID() = default;

void PID::setGoal(Eigen::Vector3d& goal_xyyaw_in_odom)
{
    goal_xyyaw_in_odom_  = std::make_unique<Eigen::Vector3d>(goal_xyyaw_in_odom);
}

void PID::getControl(const Eigen::Vector3d& curr_xyyaw_in_odom,
                                const double& dt,
                                Eigen::Vector2d& control_cmd)
{
    // Compute the difference between the first two elements of the vectors
    Eigen::Vector2d dpos = (goal_xyyaw_in_odom_->head<2>() - curr_xyyaw_in_odom.head<2>());

    double yaw = curr_xyyaw_in_odom[2];
    Eigen::Matrix2d odom2baselink;
    odom2baselink << cos(yaw), sin(yaw),
                        -sin(yaw), cos(yaw); 

    Eigen::Vector2d diff_in_base = odom2baselink * dpos;

    double dx = diff_in_base[0];
    double dy = diff_in_base[1];
    double dist = sqrt(dx * dx + dy * dy);

    // get the linear velocity
    double v_Kp = 0.8;
    double lin_vel = v_Kp*(dx/dt);

    // get the angular velocity and normalize it if needed
    double dyaw = atan2(dy, dx);

    double rotate_dist_threshold = 0.01;

    if (dist < rotate_dist_threshold) {
        dyaw = (*goal_xyyaw_in_odom_)[2] - curr_xyyaw_in_odom[2];
        dyaw = normalize_angle(dyaw);
    }


    double yaw_Kp = 0.5;
    double angular_vel = yaw_Kp * dyaw;

    // return linear_vel, angular_vel
    control_cmd << lin_vel, angular_vel; 

}


} /* namespace */