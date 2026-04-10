#ifndef NEUPAN_CONTROLLER__NRMP_OPTIMIZER_HPP_
#define NEUPAN_CONTROLLER__NRMP_OPTIMIZER_HPP_

#include <Eigen/Dense>
#include <vector>
#include "neupan_controller/diff_drive_model.hpp"

namespace neupan_controller
{

struct NRMPParams
{
  int horizon = 10;
  double dt = 0.1;
  int iterations = 3;
  double q_pos = 5.0;
  double q_theta = 2.0;
  double r_v = 1.0;
  double r_omega = 0.5;
  double d_safe = 0.35;
  double collision_weight = 50.0;
  double v_max = 0.6;
  double v_min = -0.2;
  double omega_max = 0.8;
  double step_size = 0.05;

  // Proximal coefficient (from original NeuPAN paper)
  // Penalizes trajectory deviation from previous iteration: bk * ||u - u_prev||^2
  // This is the key convergence guarantee that prevents oscillation
  double bk = 0.1;              ///< Proximal coefficient (higher = more damping, slower convergence)
};

struct NRMPResult
{
  Eigen::MatrixXd controls;
  Eigen::MatrixXd trajectory;
  double cost;
  int iterations_run;
};

class NRMPOptimizer
{
public:
  NRMPOptimizer() = default;
  void configure(const NRMPParams & params);
  NRMPResult solve(
    const State & x0,
    const Eigen::MatrixXd & reference,
    const std::vector<Eigen::Vector2d> & obstacles,
    const Eigen::MatrixXd * warm_start = nullptr);

private:
  NRMPParams params_;

  /// OSQP-based QP solver (matching Python cvxpylayers/ECOS)
  NRMPResult solve_osqp(
    const State & x0,
    const Eigen::MatrixXd & reference,
    const std::vector<Eigen::Vector2d> & obstacles,
    const Eigen::MatrixXd * warm_start);

  /// Gradient descent fallback (when OSQP not available)
  NRMPResult solve_gradient(
    const State & x0,
    const Eigen::MatrixXd & reference,
    const std::vector<Eigen::Vector2d> & obstacles,
    const Eigen::MatrixXd * warm_start);

  double compute_cost(
    const Eigen::MatrixXd & trajectory,
    const Eigen::MatrixXd & reference,
    const Eigen::MatrixXd & controls,
    const std::vector<Eigen::Vector2d> & obstacles) const;
  Eigen::Vector3d collision_gradient(
    const Eigen::Vector3d & state,
    const std::vector<Eigen::Vector2d> & obstacles) const;
  void clamp_control(Control & u) const;
};

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__NRMP_OPTIMIZER_HPP_
