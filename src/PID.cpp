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

template<typename T>
T clip(const T& value, const T& min_value, const T& max_value) {
    return std::max(min_value, std::min(max_value, value));
}


namespace wfc {

    PID::PID( ) {
        goal_xyyaw_in_odom_ = std::make_unique<Eigen::Vector3d>();
        Eigen::Vector3d start({0.,0.,0.});
        this->setGoal(start);
    }

    PID::PID(double& lin_Kp , double& lin_vel_max, double& lin_vel_min,
            double& ang_Kp, double& ang_vel_max, double& ang_vel_min,
            double rotate_dist_threshold
        ) {
        goal_xyyaw_in_odom_ = std::make_unique<Eigen::Vector3d>();
        Eigen::Vector3d start({0.,0.,0.});
        this->setGoal(start);

        lin_Kp_ = lin_Kp;
        lin_vel_max_ = lin_vel_max;
        lin_vel_min_ = lin_vel_min;

        ang_Kp_ = ang_Kp;
        ang_vel_max_ = ang_vel_max;
        ang_vel_min_ = ang_vel_min;



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
        double lin_vel = lin_Kp_*(dx/dt);

        // get the angular velocity and normalize it if needed
        double dyaw = atan2(dy, dx);

        double rotate_dist_threshold = 0.05;

        if (dist < rotate_dist_threshold) {
            dyaw = (*goal_xyyaw_in_odom_)[2] - curr_xyyaw_in_odom[2];
            dyaw = normalize_angle(dyaw);
        }

        double angular_vel = ang_Kp_ * (dyaw/dt);

        // clip velocities
        lin_vel = clip(lin_vel, lin_vel_min_, lin_vel_max_);
        angular_vel = clip(angular_vel, ang_vel_min_, ang_vel_max_);

        // return linear_vel, angular_vel
        control_cmd << lin_vel, angular_vel; 

    }


} /* namespace */