#include <gtest/gtest.h>
#include "neupan_controller/diff_drive_model.hpp"

using namespace neupan_controller;

TEST(DiffDriveModel, StraightLine)
{
  State x0(0.0, 0.0, 0.0);
  Control u(1.0, 0.0);
  double dt = 1.0;

  State x1 = DiffDriveModel::propagate(x0, u, dt);
  EXPECT_NEAR(x1(0), 1.0, 1e-9);
  EXPECT_NEAR(x1(1), 0.0, 1e-9);
  EXPECT_NEAR(x1(2), 0.0, 1e-9);
}

TEST(DiffDriveModel, PureTurn)
{
  State x0(0.0, 0.0, 0.0);
  Control u(0.0, M_PI / 2.0);
  double dt = 1.0;

  State x1 = DiffDriveModel::propagate(x0, u, dt);
  EXPECT_NEAR(x1(0), 0.0, 1e-9);
  EXPECT_NEAR(x1(1), 0.0, 1e-9);
  EXPECT_NEAR(x1(2), M_PI / 2.0, 1e-9);
}

TEST(DiffDriveModel, DiagonalMotion)
{
  State x0(0.0, 0.0, M_PI / 4.0);
  Control u(1.0, 0.0);
  double dt = 1.0;

  State x1 = DiffDriveModel::propagate(x0, u, dt);
  EXPECT_NEAR(x1(0), std::cos(M_PI / 4.0), 1e-9);
  EXPECT_NEAR(x1(1), std::sin(M_PI / 4.0), 1e-9);
  EXPECT_NEAR(x1(2), M_PI / 4.0, 1e-9);
}

TEST(DiffDriveModel, TrajectoryPropagation)
{
  State x0(0.0, 0.0, 0.0);
  int N = 5;
  Eigen::MatrixXd controls(2, N);
  controls.setZero();
  controls.row(0).setConstant(0.5);
  double dt = 0.1;

  Eigen::MatrixXd traj = DiffDriveModel::propagate_trajectory(x0, controls, dt);
  ASSERT_EQ(traj.cols(), N + 1);
  ASSERT_EQ(traj.rows(), 3);

  EXPECT_NEAR(traj(0, N), 0.25, 1e-9);
  EXPECT_NEAR(traj(1, N), 0.0, 1e-9);
}

TEST(DiffDriveModel, JacobianState)
{
  State x(1.0, 2.0, M_PI / 6.0);
  Control u(0.5, 0.1);
  double dt = 0.1;

  auto J = DiffDriveModel::jacobian_state(x, u, dt);
  EXPECT_NEAR(J(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(J(0, 1), 0.0, 1e-9);
  EXPECT_NEAR(J(0, 2), -0.5 * std::sin(M_PI / 6.0) * 0.1, 1e-9);
  EXPECT_NEAR(J(2, 0), 0.0, 1e-9);
  EXPECT_NEAR(J(2, 1), 0.0, 1e-9);
  EXPECT_NEAR(J(2, 2), 1.0, 1e-9);
}

TEST(DiffDriveModel, JacobianControl)
{
  State x(0.0, 0.0, M_PI / 3.0);
  Control u(0.5, 0.1);
  double dt = 0.1;

  auto B = DiffDriveModel::jacobian_control(x, u, dt);
  EXPECT_NEAR(B(0, 0), std::cos(M_PI / 3.0) * dt, 1e-9);
  EXPECT_NEAR(B(0, 1), 0.0, 1e-9);
  EXPECT_NEAR(B(1, 0), std::sin(M_PI / 3.0) * dt, 1e-9);
  EXPECT_NEAR(B(2, 1), dt, 1e-9);
}
