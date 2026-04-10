#include "neupan_controller/obstacle_extractor.hpp"
#include <algorithm>
#include <cmath>

namespace neupan_controller
{

std::vector<Eigen::Vector2d> ObstacleExtractor::extract(
  const nav2_costmap_2d::Costmap2D & costmap,
  double robot_x, double robot_y,
  double radius,
  unsigned char cost_threshold,
  size_t max_points)
{
  std::vector<Eigen::Vector2d> obstacles;
  obstacles.reserve(max_points);

  const double resolution = costmap.getResolution();
  const double origin_x = costmap.getOriginX();
  const double origin_y = costmap.getOriginY();
  const unsigned int size_x = costmap.getSizeInCellsX();
  const unsigned int size_y = costmap.getSizeInCellsY();

  unsigned int min_mx, min_my, max_mx, max_my;
  int raw_min_mx = static_cast<int>((robot_x - radius - origin_x) / resolution);
  int raw_min_my = static_cast<int>((robot_y - radius - origin_y) / resolution);
  int raw_max_mx = static_cast<int>((robot_x + radius - origin_x) / resolution) + 1;
  int raw_max_my = static_cast<int>((robot_y + radius - origin_y) / resolution) + 1;

  min_mx = static_cast<unsigned int>(std::max(0, raw_min_mx));
  min_my = static_cast<unsigned int>(std::max(0, raw_min_my));
  max_mx = static_cast<unsigned int>(std::min(static_cast<int>(size_x), raw_max_mx));
  max_my = static_cast<unsigned int>(std::min(static_cast<int>(size_y), raw_max_my));

  const double radius_sq = radius * radius;

  for (unsigned int my = min_my; my < max_my; ++my) {
    for (unsigned int mx = min_mx; mx < max_mx; ++mx) {
      unsigned char cost = costmap.getCost(mx, my);
      if (cost < cost_threshold) {
        continue;
      }

      double wx = origin_x + (static_cast<double>(mx) + 0.5) * resolution;
      double wy = origin_y + (static_cast<double>(my) + 0.5) * resolution;

      double dx = wx - robot_x;
      double dy = wy - robot_y;
      if (dx * dx + dy * dy <= radius_sq) {
        obstacles.emplace_back(wx, wy);
        if (obstacles.size() >= max_points) {
          return obstacles;
        }
      }
    }
  }

  return obstacles;
}

}  // namespace neupan_controller
