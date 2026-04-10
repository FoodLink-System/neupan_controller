#include <gtest/gtest.h>
#include "neupan_controller/nrmp_optimizer.hpp"

using namespace neupan_controller;

class NRMPOptimizerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_.horizon = 10;
    params_.dt = 0.1;
    params_.iterations = 20;
    params_.q_pos = 5.0;
    params_.q_theta = 2.0;
    params_.r_v = 1.0;
    params_.r_omega = 0.5;
    params_.d_safe = 0.35;
    params_.collision_weight = 50.0;
    params_.v_max = 0.6;
    params_.v_min = -0.2;
    params_.omega_max = 0.8;
    params_.step_size = 0.1;
    optimizer_.configure(params_);
  }

  NRMPParams params_;
  NRMPOptimizer optimizer_;
};

TEST_F(NRMPOptimizerTest, AtGoalProducesZeroVelocity)
{
  State x0(1.0, 0.0, 0.0);
  Eigen::MatrixXd ref(3, params_.horizon + 1);
  for (int k = 0; k <= params_.horizon; ++k) {
    ref.col(k) = x0;
  }

  std::vector<Eigen::Vector2d> obstacles;
  auto result = optimizer_.solve(x0, ref, obstacles);

  EXPECT_NEAR(result.controls(0, 0), 0.0, 0.05);
  EXPECT_NEAR(result.controls(1, 0), 0.0, 0.05);
}

TEST_F(NRMPOptimizerTest, StraightLineTracking)
{
  State x0(0.0, 0.0, 0.0);
  Eigen::MatrixXd ref(3, params_.horizon + 1);
  for (int k = 0; k <= params_.horizon; ++k) {
    ref(0, k) = 0.5 * k * params_.dt;
    ref(1, k) = 0.0;
    ref(2, k) = 0.0;
  }

  std::vector<Eigen::Vector2d> obstacles;
  auto result = optimizer_.solve(x0, ref, obstacles);

  EXPECT_GT(result.controls(0, 0), 0.01);  // positive forward velocity
  EXPECT_NEAR(result.controls(1, 0), 0.0, 0.1);
  EXPECT_EQ(result.controls.cols(), params_.horizon);
  EXPECT_EQ(result.trajectory.cols(), params_.horizon + 1);
}

TEST_F(NRMPOptimizerTest, ObstacleAvoidance)
{
  State x0(0.0, 0.0, 0.0);
  Eigen::MatrixXd ref(3, params_.horizon + 1);
  for (int k = 0; k <= params_.horizon; ++k) {
    ref(0, k) = 0.3 * k * params_.dt;
    ref(1, k) = 0.0;
    ref(2, k) = 0.0;
  }

  std::vector<Eigen::Vector2d> obstacles;
  obstacles.emplace_back(0.2, 0.0);

  auto result = optimizer_.solve(x0, ref, obstacles);
  EXPECT_GT(result.cost, 0.0);
}

TEST_F(NRMPOptimizerTest, VelocityClamping)
{
  State x0(0.0, 0.0, 0.0);
  Eigen::MatrixXd ref(3, params_.horizon + 1);
  for (int k = 0; k <= params_.horizon; ++k) {
    ref(0, k) = 10.0 * k * params_.dt;
    ref(1, k) = 0.0;
    ref(2, k) = 0.0;
  }

  std::vector<Eigen::Vector2d> obstacles;
  auto result = optimizer_.solve(x0, ref, obstacles);

  for (int k = 0; k < params_.horizon; ++k) {
    EXPECT_LE(result.controls(0, k), params_.v_max + 1e-9);
    EXPECT_GE(result.controls(0, k), params_.v_min - 1e-9);
    EXPECT_LE(std::abs(result.controls(1, k)), params_.omega_max + 1e-9);
  }
}

TEST_F(NRMPOptimizerTest, WarmStartImprovesCost)
{
  State x0(0.0, 0.0, 0.0);
  Eigen::MatrixXd ref(3, params_.horizon + 1);
  for (int k = 0; k <= params_.horizon; ++k) {
    ref(0, k) = 0.5 * k * params_.dt;
    ref(1, k) = 0.0;
    ref(2, k) = 0.0;
  }
  std::vector<Eigen::Vector2d> obstacles;

  auto result1 = optimizer_.solve(x0, ref, obstacles);

  Eigen::MatrixXd warm(2, params_.horizon);
  warm.leftCols(params_.horizon - 1) = result1.controls.rightCols(params_.horizon - 1);
  warm.col(params_.horizon - 1).setZero();

  auto result2 = optimizer_.solve(x0, ref, obstacles, &warm);
  EXPECT_LE(result2.cost, result1.cost + 1e-6);
}
