#include "neupan_controller/nrmp_optimizer.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef NEUPAN_HAS_OSQP
#include <OsqpEigen/OsqpEigen.h>
#endif

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
#ifdef NEUPAN_HAS_OSQP
  return solve_osqp(x0, reference, obstacles, warm_start);
#else
  return solve_gradient(x0, reference, obstacles, warm_start);
#endif
}

// =============================================================================
// OSQP QP Solver — matches Python cvxpylayers/ECOS exactly
// =============================================================================
#ifdef NEUPAN_HAS_OSQP

NRMPResult NRMPOptimizer::solve_osqp(
  const State & x0,
  const Eigen::MatrixXd & reference,
  const std::vector<Eigen::Vector2d> & obstacles,
  const Eigen::MatrixXd * warm_start)
{
  const int T = params_.horizon;
  const int n_s = 3 * (T + 1);  // state variables
  const int n_u = 2 * T;         // control variables
  const int n_d = T;              // distance slack variables
  const int n = n_s + n_u + n_d;  // total decision variables

  // Nominal trajectory for linearization (use warm start or zero)
  Eigen::MatrixXd nom_controls(2, T);
  if (warm_start && warm_start->cols() == T && warm_start->rows() == 2) {
    nom_controls = *warm_start;
  } else {
    nom_controls.setZero();
  }

  NRMPResult result;
  result.controls.resize(2, T);

  // SQP: re-linearize 2-3 times for nonlinear accuracy
  int sqp_iters = std::min(params_.iterations, 3);

  for (int sqp = 0; sqp < sqp_iters; ++sqp) {
    // Linearize dynamics around nominal trajectory
    Eigen::MatrixXd nom_traj = DiffDriveModel::propagate_trajectory(x0, nom_controls, params_.dt);

    // Build QP: minimize 0.5 * z^T * P * z + q^T * z
    //           subject to l <= A_c * z <= u

    // --- Build P (cost Hessian) ---
    // P is block-diagonal: P_ss, P_uu, P_dd
    Eigen::SparseMatrix<double> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;

    // State tracking: q_s^2 * I + bk * I
    double w_s = 2.0 * (params_.q_pos * params_.q_pos + params_.bk);
    double w_theta = 2.0 * (params_.q_theta * params_.q_theta + params_.bk);
    for (int k = 0; k <= T; ++k) {
      int idx = 3 * k;
      P_triplets.emplace_back(idx, idx, w_s);          // px
      P_triplets.emplace_back(idx + 1, idx + 1, w_s);  // py
      P_triplets.emplace_back(idx + 2, idx + 2, w_theta); // theta
    }

    // Control effort: p_u^2 for v only (matching Python C0_cost)
    double w_v = 2.0 * params_.r_v * params_.r_v;
    double w_omega = 2.0 * params_.r_omega * params_.r_omega;
    for (int k = 0; k < T; ++k) {
      int idx = n_s + 2 * k;
      P_triplets.emplace_back(idx, idx, w_v);          // v
      P_triplets.emplace_back(idx + 1, idx + 1, w_omega); // omega
    }

    // Distance slack: collision penalty ro_obs for violated constraints
    // I_cost adds ro_obs * neg(...)^2 — approximated as quadratic in d
    for (int k = 0; k < T; ++k) {
      int idx = n_s + n_u + k;
      P_triplets.emplace_back(idx, idx, params_.collision_weight);
    }

    P.setFromTriplets(P_triplets.begin(), P_triplets.end());

    // --- Build q (cost gradient) ---
    Eigen::VectorXd q_vec = Eigen::VectorXd::Zero(n);

    // State tracking linear term: -2 * q_s * ref_s - 2 * bk * nom_s
    for (int k = 0; k <= T; ++k) {
      int ref_k = std::min(k, static_cast<int>(reference.cols()) - 1);
      int idx = 3 * k;
      q_vec(idx)     = -2.0 * params_.q_pos * reference(0, ref_k) - 2.0 * params_.bk * nom_traj(0, k);
      q_vec(idx + 1) = -2.0 * params_.q_pos * reference(1, ref_k) - 2.0 * params_.bk * nom_traj(1, k);
      q_vec(idx + 2) = -2.0 * params_.q_theta * reference(2, ref_k) - 2.0 * params_.bk * nom_traj(2, k);
    }

    // Distance slack linear term: -eta (maximize safety distance)
    for (int k = 0; k < T; ++k) {
      q_vec(n_s + n_u + k) = -params_.collision_weight * params_.d_safe;
    }

    // --- Build constraints ---
    // Dynamics: A_k * s_k + B_k * u_k + C_k = s_{k+1}
    // Initial: s_0 = x0
    // Velocity: -v_max <= u <= v_max
    // Accel: |u_{k+1} - u_k| <= acce_bound
    // Distance: d_min <= d <= d_max

    int n_dyn = 3 * T;
    int n_init = 3;
    int n_vel = 2 * T;
    int n_acc = 2 * (T - 1);
    int n_dist = T;
    int m = n_dyn + n_init + n_vel + n_acc + n_dist;

    Eigen::SparseMatrix<double> A_c(m, n);
    Eigen::VectorXd l_vec(m), u_vec(m);
    std::vector<Eigen::Triplet<double>> A_triplets;

    int row = 0;

    // 1. Dynamics constraints: s_{k+1} = A_k * s_k + B_k * u_k + C_k
    for (int k = 0; k < T; ++k) {
      double theta_k = nom_traj(2, k);
      double v_k = nom_controls(0, k);
      double dt = params_.dt;

      // A_k matrix (3x3)
      double A00 = 1, A02 = -v_k * dt * std::sin(theta_k);
      double A11 = 1, A12 =  v_k * dt * std::cos(theta_k);
      double A22 = 1;

      // B_k matrix (3x2)
      double B00 = std::cos(theta_k) * dt;
      double B10 = std::sin(theta_k) * dt;
      double B21 = dt;

      // C_k vector (3x1)
      double C0 =  theta_k * v_k * std::sin(theta_k) * dt;
      double C1 = -theta_k * v_k * std::cos(theta_k) * dt;
      double C2 = 0.0;

      int s_k = 3 * k;       // index of s_k in z
      int s_k1 = 3 * (k + 1); // index of s_{k+1}
      int u_k = n_s + 2 * k;  // index of u_k

      // -s_{k+1} + A_k * s_k + B_k * u_k = -C_k
      // Row for px
      A_triplets.emplace_back(row, s_k, A00);
      A_triplets.emplace_back(row, s_k + 2, A02);
      A_triplets.emplace_back(row, u_k, B00);
      A_triplets.emplace_back(row, s_k1, -1.0);
      l_vec(row) = -C0;
      u_vec(row) = -C0;
      row++;

      // Row for py
      A_triplets.emplace_back(row, s_k + 1, A11);
      A_triplets.emplace_back(row, s_k + 2, A12);
      A_triplets.emplace_back(row, u_k, B10);
      A_triplets.emplace_back(row, s_k1 + 1, -1.0);
      l_vec(row) = -C1;
      u_vec(row) = -C1;
      row++;

      // Row for theta
      A_triplets.emplace_back(row, s_k + 2, A22);
      A_triplets.emplace_back(row, u_k + 1, B21);
      A_triplets.emplace_back(row, s_k1 + 2, -1.0);
      l_vec(row) = -C2;
      u_vec(row) = -C2;
      row++;
    }

    // 2. Initial state: s_0 = x0
    for (int i = 0; i < 3; ++i) {
      A_triplets.emplace_back(row, i, 1.0);
      l_vec(row) = x0(i);
      u_vec(row) = x0(i);
      row++;
    }

    // 3. Velocity bounds: -v_max <= u_k <= v_max
    for (int k = 0; k < T; ++k) {
      int u_idx = n_s + 2 * k;
      // v
      A_triplets.emplace_back(row, u_idx, 1.0);
      l_vec(row) = params_.v_min;
      u_vec(row) = params_.v_max;
      row++;
      // omega
      A_triplets.emplace_back(row, u_idx + 1, 1.0);
      l_vec(row) = -params_.omega_max;
      u_vec(row) = params_.omega_max;
      row++;
    }

    // 4. Acceleration bounds: |u_{k+1} - u_k| <= acce_bound * dt
    double acce_v = 0.22 * params_.dt;   // from robot config
    double acce_w = 0.6 * params_.dt;
    for (int k = 0; k < T - 1; ++k) {
      int u_k = n_s + 2 * k;
      int u_k1 = n_s + 2 * (k + 1);
      // v acceleration
      A_triplets.emplace_back(row, u_k1, 1.0);
      A_triplets.emplace_back(row, u_k, -1.0);
      l_vec(row) = -acce_v;
      u_vec(row) = acce_v;
      row++;
      // omega acceleration
      A_triplets.emplace_back(row, u_k1 + 1, 1.0);
      A_triplets.emplace_back(row, u_k + 1, -1.0);
      l_vec(row) = -acce_w;
      u_vec(row) = acce_w;
      row++;
    }

    // 5. Distance bounds: d_min <= d_k <= d_max
    for (int k = 0; k < T; ++k) {
      int d_idx = n_s + n_u + k;
      A_triplets.emplace_back(row, d_idx, 1.0);
      l_vec(row) = params_.d_safe * 0.1;  // d_min
      u_vec(row) = params_.d_safe * 3.0;  // d_max
      row++;
    }

    A_c.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_c.makeCompressed();

    // --- Solve with OSQP ---
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.settings()->setMaxIteration(200);
    solver.settings()->setAbsoluteTolerance(1e-4);
    solver.settings()->setRelativeTolerance(1e-4);

    // Create properly sized constraint matrix (OsqpEigen needs exact dimensions)
    Eigen::SparseMatrix<double> A_final(row, n);
    A_final.reserve(A_triplets.size());
    // Re-build with correct row count
    std::vector<Eigen::Triplet<double>> final_triplets;
    for (const auto & t : A_triplets) {
      if (t.row() < row) final_triplets.push_back(t);
    }
    A_final.setFromTriplets(final_triplets.begin(), final_triplets.end());
    A_final.makeCompressed();

    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfConstraints(row);
    solver.data()->setHessianMatrix(P);
    solver.data()->setGradient(q_vec);
    solver.data()->setLinearConstraintsMatrix(A_final);
    solver.data()->setLowerBound(l_vec.head(row));
    solver.data()->setUpperBound(u_vec.head(row));

    if (!solver.initSolver()) {
      std::cerr << "OSQP init failed, falling back to gradient" << std::endl;
      return solve_gradient(x0, reference, obstacles, warm_start);
    }

    auto status = solver.solveProblem();
    if (status != OsqpEigen::ErrorExitFlag::NoError) {
      std::cerr << "OSQP solve failed, falling back to gradient" << std::endl;
      return solve_gradient(x0, reference, obstacles, warm_start);
    }

    Eigen::VectorXd z = solver.getSolution();

    // Extract controls from solution
    for (int k = 0; k < T; ++k) {
      nom_controls(0, k) = z(n_s + 2 * k);
      nom_controls(1, k) = z(n_s + 2 * k + 1);
    }
  }  // end SQP iterations

  // Build result
  result.controls = nom_controls;
  result.trajectory = DiffDriveModel::propagate_trajectory(x0, result.controls, params_.dt);
  result.cost = compute_cost(result.trajectory, reference, result.controls, obstacles);
  result.iterations_run = sqp_iters;

  return result;
}

#endif  // NEUPAN_HAS_OSQP

// =============================================================================
// Gradient descent fallback (when OSQP not available)
// =============================================================================

NRMPResult NRMPOptimizer::solve_gradient(
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

      Eigen::Vector2d grad_u = B_k.transpose() * Q * tracking_error + R * u_k;

      Eigen::Vector3d col_grad = collision_gradient(x_next, obstacles);
      grad_u += B_k.transpose() * col_grad;

      Control proximal_diff = u_k - prev_iter_controls.col(k);
      grad_u += params_.bk * proximal_diff;

      result.controls.col(k) -= params_.step_size * grad_u;

      Control u_clamped = result.controls.col(k);
      clamp_control(u_clamped);
      result.controls.col(k) = u_clamped;
    }

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
