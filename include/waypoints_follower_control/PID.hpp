#pragma once

#include <memory>
#include <Eigen/Core>
#include <iostream>
using namespace std;

namespace wfc {

/*!
 * \brief Class representing a Proportional-Integral-Derivative (PID) controller.
 *
 * This class provides the algorithmic implementation of a PID controller used
 * to compute control commands for a robotic system based on its current state
 * and desired goals.
 */
class PID
{
    public:

    /*!
    * \brief Default constructor.
    *
    * Initializes a PID controller with default parameters.
    */
    PID();

    /*!
    * \brief Parameterized constructor.
    *
    * Initializes a PID controller with the specified gains and limits for both linear and angular velocity.
    * \param lin_Kp Proportional gain for the linear velocity control.
    * \param lin_Ki Integral gain for the linear velocity control.
    * \param lin_Kd Derivative gain for the linear velocity control.
    * \param lin_vel_max Maximum linear velocity.
    * \param lin_vel_min Minimum linear velocity.

    * \param ang_Kp Derivative gain for the angular velocity control.
    * \param ang_Ki Integral gain for the angular velocity control.
    * \param ang_Kd Derivative gain for the angular velocity control.
    * \param ang_vel_max Maximum angular velocity.
    * \param ang_vel_min Minimum angular velocity.
    * \param rotate_dist_threshold Distance threshold for initiating rotational control.
    */
    PID(double& lin_Kp, double& lin_Ki, double& lin_Kd,double& lin_vel_max, double& lin_vel_min,
        double& ang_Kp, double& ang_Ki, double& ang_Kd, double& ang_vel_max, double& ang_vel_min,
        double& rotate_dist_threshold);

    /*!
    * \brief Destructor.
    */
    virtual ~PID();

    /*!
    * \brief Sets a new goal position and orientation for the controller.
    *
    * The goal is given as a 3D vector, where the first two components are the x and y coordinates,
    * and the third component is the yaw angle.
    * \param data A 3D vector representing the target position and orientation.
    */
    void setGoal(Eigen::Vector3d& data);

    /*!
    * \brief Computes the control command based on the current position and orientation.
    *
    * Calculates the appropriate linear and angular velocity commands to reach the goal.
    * \param curr_xyyaw_in_odom A 3D vector representing the current x, y coordinates and yaw angle in the odom frame.
    * \param dt Time step between control updates.
    * \param contrl_cmd A 2D vector to store the resulting control command (linear velocity, angular velocity).
    */
    void getControl(const Eigen::Vector3d& curr_xyyaw_in_odom,
                    const double& dt,
                    Eigen::Vector2d& contrl_cmd);

    void setControllerParams(double& lin_Kp, double& lin_Ki, double& lin_Kd,
        double& ang_Kp, double& ang_Ki, double& ang_Kd,
        double& rotate_dist_threshold);

    private:

    //! The desired goal position and orientation in the odom frame.
    std::unique_ptr<Eigen::Vector3d> goal_xyyaw_in_odom_;

    //! Proportional gain for the angular velocity control.
    double ang_Kp_;
    double ang_Ki_;
    double ang_Kd_;
    double yaw_integral_error_;
    double prev_yaw_error_;

    //! Proportional gain for the linear velocity control.
    double lin_Kp_;
    double lin_Ki_;
    double lin_Kd_;
    double integral_error_;
    double prev_error_;

    //! Maximum linear velocity.
    double lin_vel_max_;
    //! Minimum linear velocity.
    double lin_vel_min_;

    //! Maximum angular velocity.
    double ang_vel_max_;
    //! Minimum angular velocity.
    double ang_vel_min_;

    //! Distance threshold to initiate rotation towards the goal.
    double rotate_dist_threshold_;


};

} /* namespace */
