// ros2_src/neupan_controller/include/neupan_controller/neupan_controller.hpp
#ifndef NEUPAN_CONTROLLER__NEUPAN_CONTROLLER_HPP_
#define NEUPAN_CONTROLLER__NEUPAN_CONTROLLER_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nav2_core/controller.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <tf2_ros/buffer.h>

#include "neupan_controller/nrmp_optimizer.hpp"
#include "neupan_controller/dune_inference.hpp"

namespace neupan_controller
{

class NeuPANController : public nav2_core::Controller
{
public:
  NeuPANController() = default;
  ~NeuPANController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string plugin_name_;
  rclcpp::Logger logger_{rclcpp::get_logger("NeuPANController")};
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;

  nav_msgs::msg::Path global_plan_;

  NRMPOptimizer optimizer_;
  NRMPParams params_;
  DUNEInference dune_;
  std::string dune_model_path_;

  // Warm start
  Eigen::MatrixXd prev_controls_;
  bool has_warm_start_{false};

  // Speed limit
  double speed_limit_{0.0};
  bool speed_limit_is_percentage_{false};

  // Publisher for local plan visualization
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_plan_pub_;

  // Direct scan subscribers — raw obstacle points, no costmap
  std::vector<rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr> scan_subs_;
  std::mutex scan_mutex_;
  std::vector<Eigen::Vector2d> latest_scan_points_;  // in world frame

  // Obstacle persistence — keep recent scan points for a short window
  // so obstacles behind the robot aren't instantly forgotten
  struct TimedPoints {
    std::vector<Eigen::Vector2d> points;  // world frame
    double timestamp;
  };
  std::deque<TimedPoints> scan_history_;
  double obstacle_persistence_sec_{0.0};  // 0 = no persistence, use latest scan only

  // Scan parameters
  std::vector<std::string> scan_topics_;
  double scan_range_max_{5.0};
  size_t max_obstacle_points_{200};
  double desired_speed_{0.4};

  /// Callback for LaserScan messages — converts to obstacle points in base_link frame.
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg,
                    const std::string & frame_id);

  /// Get current obstacle points (thread-safe copy).
  std::vector<Eigen::Vector2d> getObstaclePoints();

  /// Transform global plan to robot frame and prune behind robot.
  nav_msgs::msg::Path transformGlobalPlan(
    const geometry_msgs::msg::PoseStamped & pose);

  /// Publish the optimized local trajectory for visualization.
  void publishLocalPlan(const Eigen::MatrixXd & trajectory);
};

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__NEUPAN_CONTROLLER_HPP_
