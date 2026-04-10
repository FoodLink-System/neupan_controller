#ifndef NEUPAN_CONTROLLER__OBSTACLE_EXTRACTOR_HPP_
#define NEUPAN_CONTROLLER__OBSTACLE_EXTRACTOR_HPP_

#include <Eigen/Dense>
#include <vector>
#include <nav2_costmap_2d/costmap_2d.hpp>

namespace neupan_controller
{

class ObstacleExtractor
{
public:
  static std::vector<Eigen::Vector2d> extract(
    const nav2_costmap_2d::Costmap2D & costmap,
    double robot_x, double robot_y,
    double radius,
    unsigned char cost_threshold = 253,
    size_t max_points = 200);
};

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__OBSTACLE_EXTRACTOR_HPP_
