#include "waypoints_follower_control/PID.hpp"
#include <ros/ros.h>
#include <utility>

using namespace std;


/*!
 * \brief Normalizes an angle to the range of -π to π.
 *
 * This function ensures that an input angle, provided in radians, falls within
 * the range of -π to π by adjusting it as necessary.
 * \param yaw The input angle in radians.
 * \return The normalized angle in radians, constrained to the range of -π to π.
 */
double clip_angle(double yaw) {
    // Normalize the angle between -π and π
    if (yaw > M_PI) {
        yaw -= 2 * M_PI;
    }
    else if (yaw < -M_PI) {
        yaw += 2 * M_PI;
    }
    else {
        return yaw;
    }

    return clip_angle(yaw);
}


template <typename T> 
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/*!
 * \brief Clips a value to fall within the specified range.
 *
 * This function ensures that a value is within the given minimum and maximum limits.
 * If the value is less than the minimum, it returns the minimum. If greater than the
 * maximum, it returns the maximum. Otherwise, it returns the value itself.
 * 
 * \tparam T The data type of the value, which should be comparable.
 * \param value The input value to be clipped.
 * \param min_value The minimum allowed value.
 * \param max_value The maximum allowed value.
 * \return The clipped value constrained within the range [min_value, max_value].
 */
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

    PID::PID(double& lin_Kp, double& lin_Ki, double& lin_Kd,double& lin_vel_max, double& lin_vel_min,
        double& ang_Kp, double& ang_Ki, double& ang_Kd, double& ang_vel_max, double& ang_vel_min,
        double& rotate_dist_threshold):
        lin_Kp_(lin_Kp), lin_Ki_(lin_Ki), lin_Kd_(lin_Kd),lin_vel_max_(lin_vel_max), lin_vel_min_(lin_vel_min),
        integral_error_(0.),prev_error_(0.),
        ang_Kp_(ang_Kp), ang_Ki_(ang_Ki), ang_Kd_(ang_Kd),ang_vel_max_(ang_vel_max), ang_vel_min_(ang_vel_min),
        yaw_integral_error_(0),prev_yaw_error_(0.),
        rotate_dist_threshold_(rotate_dist_threshold)
        {
        goal_xyyaw_in_odom_ = std::make_unique<Eigen::Vector3d>();
        Eigen::Vector3d start({0.,0.,0.});
        this->setGoal(start);
    }



    PID::~PID() = default;

    void PID::setGoal(Eigen::Vector3d& goal_xyyaw_in_odom)
    {
        goal_xyyaw_in_odom_  = std::make_unique<Eigen::Vector3d>(goal_xyyaw_in_odom);

    }

    void PID::setControllerParams(double& lin_Kp, double& lin_Ki, double& lin_Kd,
        double& ang_Kp, double& ang_Ki, double& ang_Kd,
        double& rotate_dist_threshold)
    {
        rotate_dist_threshold_ = rotate_dist_threshold;

        lin_Kp_ = lin_Kp;
        lin_Ki_ = lin_Ki;
        lin_Kd_ = lin_Kd;

        ang_Kp_ = ang_Kp;
        ang_Ki_ = ang_Ki;
        ang_Kd_ = ang_Kd;
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
        // ROS_INFO("dx: %f | dy: %f | dist: %f", dx,dy,dist);

        // X axis PID
        double error = dx / dt ;
        integral_error_ += error*dt;
        double derivative_error_ = (error - prev_error_)/dt;

        if (dist == 0 && error == 0){
            integral_error_ = 0;
        }

        // get the linear velocity
        // double lin_vel = lin_Kp_* error;
        double lin_vel = lin_Kp_* error + lin_Ki_ * integral_error_ + lin_Kd_ *  derivative_error_;
        prev_error_ = error;


        // get the angular velocity and normalize it if needed
        double dyaw = atan2(dy, dx);

        if (dist < rotate_dist_threshold_) {
            
            dyaw = (*goal_xyyaw_in_odom_)[2] - curr_xyyaw_in_odom[2];
            dyaw = clip_angle(dyaw);
            if (std::abs(dyaw)<=0.05){
                dyaw = 0.0;
                }
            // ROS_INFO("inside radi");
            // ROS_INFO("goal_xyyaw_in_odom_: %f | curr_xyyaw_in_odom: %f", (*goal_xyyaw_in_odom_)[2], curr_xyyaw_in_odom[2]);
            lin_vel = 0.0;
        }
        // ROS_INFO("dyaw: %f", dyaw);

        // Yaw PID
        /////////////////////////////////////// here
        // double yaw_error = static_cast<double>(sgn(dy) * (dyaw/dt));
        // ROS_INFO("dt: %f", dt);
        double yaw_error = dyaw/dt;
        // ROS_INFO("dy: %f", dy);
        // ROS_INFO("yaw_error: %f", yaw_error);
        
        yaw_integral_error_ += yaw_error*dt;
        double yaw_derivative_error_ = (yaw_error - prev_yaw_error_)/dt;

        if ((*goal_xyyaw_in_odom_)[2] == 0 && yaw_error==0){
            yaw_integral_error_ = 0;
        }

        // get the linear velocity
        // double angular_vel = ang_Kp_* error;
        double angular_vel = ang_Kp_* yaw_error + ang_Ki_ * yaw_integral_error_ + ang_Kd_ * yaw_derivative_error_;
        prev_yaw_error_ = yaw_error;
        

        


        // clip velocities
        lin_vel = clip(lin_vel, lin_vel_min_, lin_vel_max_);
        angular_vel = clip(angular_vel, ang_vel_min_, ang_vel_max_);

        // return linear_vel, angular_vel
        control_cmd << lin_vel, angular_vel; 

    }
} /* namespace */