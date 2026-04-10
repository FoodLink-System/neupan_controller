#include "neupan_controller/diff_drive_model.hpp"
#include <cmath>

namespace neupan_controller
{

State DiffDriveModel::propagate(const State & state, const Control & control, double dt)
{
  const double px = state(0);
  const double py = state(1);
  const double theta = state(2);
  const double v = control(0);
  const double omega = control(1);

  State next;
  next(0) = px + v * std::cos(theta) * dt;
  next(1) = py + v * std::sin(theta) * dt;
  next(2) = theta + omega * dt;
  return next;
}

Eigen::MatrixXd DiffDriveModel::propagate_trajectory(
  const State & x0,
  const Eigen::MatrixXd & controls,
  double dt)
{
  const int N = static_cast<int>(controls.cols());
  Eigen::MatrixXd traj(3, N + 1);
  traj.col(0) = x0;
  for (int k = 0; k < N; ++k) {
    traj.col(k + 1) = propagate(traj.col(k), controls.col(k), dt);
  }
  return traj;
}

Eigen::Matrix3d DiffDriveModel::jacobian_state(
  const State & state, const Control & control, double dt)
{
  const double theta = state(2);
  const double v = control(0);

  Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
  J(0, 2) = -v * std::sin(theta) * dt;
  J(1, 2) = v * std::cos(theta) * dt;
  return J;
}

Eigen::Matrix<double, 3, 2> DiffDriveModel::jacobian_control(
  const State & state, const Control & /*control*/, double dt)
{
  const double theta = state(2);

  Eigen::Matrix<double, 3, 2> B;
  B.setZero();
  B(0, 0) = std::cos(theta) * dt;
  B(1, 0) = std::sin(theta) * dt;
  B(2, 1) = dt;
  return B;
}

}  // namespace neupan_controller
