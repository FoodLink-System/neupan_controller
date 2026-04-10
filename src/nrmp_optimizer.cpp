#include "neupan_controller/nrmp_optimizer.hpp"
#include <cmath>
#include <algorithm>

namespace neupan_controller
{

void NRMPOptimizer::configure(const NRMPParams & params)
{
  params_ = params;
}

NRMPResult NRMPOptimizer::solve(
  const State & x0,
  const Eigen::MatrixXd & reference,
  const std::vector<Eigen::Vector2d> & obstacles,
  const Eigen::MatrixXd * warm_start)
{
  const int N = params_.horizon;
  NRMPResult result;
  result.controls.resize(2, N);

  if (warm_start && warm_start->cols() == N && warm_start->rows() == 2) {
    result.controls = *warm_start;
  } else {
    result.controls.setZero();
  }

  Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
  Q(0, 0) = params_.q_pos;
  Q(1, 1) = params_.q_pos;
  Q(2, 2) = params_.q_theta;

  Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
  R(0, 0) = params_.r_v;
  R(1, 1) = params_.r_omega;

  // Proximal alternating minimization iterations
  // Store previous iteration's controls for proximal term
  Eigen::MatrixXd prev_iter_controls = result.controls;

  for (int iter = 0; iter < params_.iterations; ++iter) {
    Eigen::MatrixXd traj = DiffDriveModel::propagate_trajectory(x0, result.controls, params_.dt);

    for (int k = 0; k < N; ++k) {
      State x_k = traj.col(k);
      Control u_k = result.controls.col(k);

      State x_next = traj.col(k + 1);
      State x_ref_next = reference.col(std::min(k + 1, static_cast<int>(reference.cols()) - 1));
      State tracking_error = x_next - x_ref_next;

      while (tracking_error(2) > M_PI) tracking_error(2) -= 2.0 * M_PI;
      while (tracking_error(2) < -M_PI) tracking_error(2) += 2.0 * M_PI;

      Eigen::Matrix<double, 3, 2> B_k = DiffDriveModel::jacobian_control(x_k, u_k, params_.dt);

      // Tracking + control effort gradient
      Eigen::Vector2d grad_u = B_k.transpose() * Q * tracking_error + R * u_k;

      // Collision avoidance gradient
      Eigen::Vector3d col_grad = collision_gradient(x_next, obstacles);
      grad_u += B_k.transpose() * col_grad;

      // Proximal term: bk * (u_k - u_k_prev_iter)
      // This is the key NeuPAN convergence mechanism — prevents oscillation
      Control proximal_diff = u_k - prev_iter_controls.col(k);
      grad_u += params_.bk * proximal_diff;

      result.controls.col(k) -= params_.step_size * grad_u;

      Control u_clamped = result.controls.col(k);
      clamp_control(u_clamped);
      result.controls.col(k) = u_clamped;
    }

    // Update previous iteration controls for next proximal step
    prev_iter_controls = result.controls;
    result.iterations_run = iter + 1;
  }

  result.trajectory = DiffDriveModel::propagate_trajectory(x0, result.controls, params_.dt);
  result.cost = compute_cost(result.trajectory, reference, result.controls, obstacles);

  return result;
}

double NRMPOptimizer::compute_cost(
  const Eigen::MatrixXd & trajectory,
  const Eigen::MatrixXd & reference,
  const Eigen::MatrixXd & controls,
  const std::vector<Eigen::Vector2d> & obstacles) const
{
  double cost = 0.0;
  const int N = static_cast<int>(controls.cols());

  for (int k = 0; k <= N; ++k) {
    int ref_k = std::min(k, static_cast<int>(reference.cols()) - 1);
    Eigen::Vector3d error = trajectory.col(k) - reference.col(ref_k);
    while (error(2) > M_PI) error(2) -= 2.0 * M_PI;
    while (error(2) < -M_PI) error(2) += 2.0 * M_PI;

    cost += params_.q_pos * (error(0) * error(0) + error(1) * error(1));
    cost += params_.q_theta * error(2) * error(2);
  }

  for (int k = 0; k < N; ++k) {
    cost += params_.r_v * controls(0, k) * controls(0, k);
    cost += params_.r_omega * controls(1, k) * controls(1, k);
  }

  for (int k = 0; k <= N; ++k) {
    Eigen::Vector2d pos(trajectory(0, k), trajectory(1, k));
    for (const auto & obs : obstacles) {
      double dist = (pos - obs).norm();
      if (dist < params_.d_safe) {
        double violation = params_.d_safe - dist;
        cost += params_.collision_weight * violation * violation;
      }
    }
  }

  return cost;
}

Eigen::Vector3d NRMPOptimizer::collision_gradient(
  const Eigen::Vector3d & state,
  const std::vector<Eigen::Vector2d> & obstacles) const
{
  Eigen::Vector3d grad = Eigen::Vector3d::Zero();
  Eigen::Vector2d pos(state(0), state(1));

  for (const auto & obs : obstacles) {
    Eigen::Vector2d diff = pos - obs;
    double dist = diff.norm();
    if (dist < params_.d_safe && dist > 1e-6) {
      double violation = params_.d_safe - dist;
      Eigen::Vector2d repulsive = -2.0 * params_.collision_weight * violation * (diff / dist);
      grad(0) += repulsive(0);
      grad(1) += repulsive(1);
    } else if (dist <= 1e-6) {
      grad(0) += params_.collision_weight * params_.d_safe * 2.0;
    }
  }

  return grad;
}

void NRMPOptimizer::clamp_control(Control & u) const
{
  u(0) = std::clamp(u(0), params_.v_min, params_.v_max);
  u(1) = std::clamp(u(1), -params_.omega_max, params_.omega_max);
}

}  // namespace neupan_controller
