#pragma once

#include <memory>
#include <math.hpp>

#include <Eigen/Core>

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
};

} /* namespace */
