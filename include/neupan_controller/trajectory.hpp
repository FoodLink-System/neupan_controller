#ifndef NEUPAN_CONTROLLER__TRAJECTORY_HPP_
#define NEUPAN_CONTROLLER__TRAJECTORY_HPP_

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/utils.h>

namespace neupan_controller
{

/// Generate a reference trajectory by sampling the global plan at dt intervals.
/// Starts from the current robot pose and walks along the path at desired_speed.
inline Eigen::MatrixXd interpolate_reference(
  const Eigen::Vector3d & current_state,
  const nav_msgs::msg::Path & path,
  int horizon,
  double dt,
  double desired_speed)
{
  Eigen::MatrixXd ref(3, horizon + 1);
  ref.col(0) = current_state;

  if (path.poses.empty()) {
    for (int k = 1; k <= horizon; ++k) {
      ref.col(k) = current_state;
    }
    return ref;
  }

  std::vector<Eigen::Vector2d> pts;
  pts.reserve(path.poses.size());
  for (const auto & pose : path.poses) {
    pts.emplace_back(pose.pose.position.x, pose.pose.position.y);
  }

  double min_dist_sq = std::numeric_limits<double>::max();
  size_t closest_idx = 0;
  Eigen::Vector2d robot_pos(current_state(0), current_state(1));
  for (size_t i = 0; i < pts.size(); ++i) {
    double d = (pts[i] - robot_pos).squaredNorm();
    if (d < min_dist_sq) {
      min_dist_sq = d;
      closest_idx = i;
    }
  }

  double accumulated = 0.0;
  size_t path_idx = closest_idx;
  double step_dist = desired_speed * dt;

  for (int k = 1; k <= horizon; ++k) {
    accumulated += step_dist;

    while (path_idx + 1 < pts.size()) {
      double seg_len = (pts[path_idx + 1] - pts[path_idx]).norm();
      if (seg_len < 1e-6) {
        ++path_idx;
        continue;
      }
      if (accumulated <= seg_len) {
        double t = accumulated / seg_len;
        Eigen::Vector2d p = pts[path_idx] + t * (pts[path_idx + 1] - pts[path_idx]);
        double theta = std::atan2(
          pts[path_idx + 1](1) - pts[path_idx](1),
          pts[path_idx + 1](0) - pts[path_idx](0));
        ref(0, k) = p(0);
        ref(1, k) = p(1);
        ref(2, k) = theta;
        accumulated = 0.0;
        break;
      }
      accumulated -= seg_len;
      ++path_idx;
    }

    if (path_idx + 1 >= pts.size()) {
      const auto & last = pts.back();
      ref(0, k) = last(0);
      ref(1, k) = last(1);
      if (path.poses.size() >= 2) {
        const auto & p2 = pts[pts.size() - 1];
        const auto & p1 = pts[pts.size() - 2];
        ref(2, k) = std::atan2(p2(1) - p1(1), p2(0) - p1(0));
      } else {
        ref(2, k) = current_state(2);
      }
    }
  }

  return ref;
}

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__TRAJECTORY_HPP_
