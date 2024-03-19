#pragma once

#include <memory>
// #include <math.hpp>

#include <Eigen/Core>

#include <iostream>
using namespace std;

namespace wfc {

/*!
 * Class containing the algorithmic part of the package.
 */

class PID
{
    public:

    /*!
    * Constructor.
    */
    PID();

    /*!
    * Constructor.
    */
    PID(double& lin_Kp , double& lin_vel_max, double& lin_vel_min,
        double& ang_Kp, double& ang_vel_max, double& ang_vel_min,
        double rotate_dist_threshold
    );


    /*!
    * Destructor.
    */
    virtual ~PID();

    /*!

    */
    void setGoal(Eigen::Vector3d& data);


    /*!

    */
    void getControl(const Eigen::Vector3d& curr_xyyaw_in_odom,
                    const double& dt,
                    Eigen::Vector2d& contrl_cmd);


    private:
    
    //! Forward declared structure that will contain the data
    //   struct Data;

    //! Pointer to data (pimpl)
    std::unique_ptr<Eigen::Vector3d> goal_xyyaw_in_odom_;

    double ang_Kp_;
    double lin_Kp_;

    double lin_vel_max_ ;
    double lin_vel_min_ ;

    double ang_vel_max_ ;
    double ang_vel_min_ ;

    double rotate_dist_threshold_;
};

} /* namespace */
