#ifndef NEUPAN_CONTROLLER__DIFF_DRIVE_MODEL_HPP_
#define NEUPAN_CONTROLLER__DIFF_DRIVE_MODEL_HPP_

#include <Eigen/Dense>

namespace neupan_controller
{

/// State vector: [px, py, theta]
using State = Eigen::Vector3d;

/// Control vector: [v, omega]
using Control = Eigen::Vector2d;

/// Differential drive kinematic model.
/// Propagates state forward by dt using Euler integration.
class DiffDriveModel
{
public:
  /// Propagate state by one timestep.
  static State propagate(const State & state, const Control & control, double dt);

  /// Propagate a trajectory of N steps from initial state with control sequence.
  static Eigen::MatrixXd propagate_trajectory(
    const State & x0,
    const Eigen::MatrixXd & controls,  // 2 x N
    double dt);

  /// Jacobian of dynamics w.r.t. state: df/dx at (state, control, dt).
  static Eigen::Matrix3d jacobian_state(const State & state, const Control & control, double dt);

  /// Jacobian of dynamics w.r.t. control: df/du at (state, control, dt).
  static Eigen::Matrix<double, 3, 2> jacobian_control(
    const State & state, const Control & control, double dt);
};

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__DIFF_DRIVE_MODEL_HPP_
