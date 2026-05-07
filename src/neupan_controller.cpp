// ros2_src/neupan_controller/src/neupan_controller.cpp
#include "neupan_controller/neupan_controller.hpp"
#include "neupan_controller/dune_inference.hpp"
#include "neupan_controller/obstacle_extractor.hpp"
#include "neupan_controller/trajectory.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <nav2_util/node_utils.hpp>
#include <nav2_util/geometry_utils.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <functional>

namespace neupan_controller
{

void NeuPANController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  // Declare and get parameters
  nav2_util::declare_parameter_if_not_declared(node, name + ".horizon", rclcpp::ParameterValue(20));
  nav2_util::declare_parameter_if_not_declared(node, name + ".dt", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(node, name + ".iterations", rclcpp::ParameterValue(50));
  nav2_util::declare_parameter_if_not_declared(node, name + ".q_pos", rclcpp::ParameterValue(30.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".q_theta", rclcpp::ParameterValue(5.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".r_v", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(node, name + ".r_omega", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(node, name + ".d_safe", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(node, name + ".collision_weight", rclcpp::ParameterValue(100.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".v_max", rclcpp::ParameterValue(0.6));
  nav2_util::declare_parameter_if_not_declared(node, name + ".v_min", rclcpp::ParameterValue(-0.2));
  nav2_util::declare_parameter_if_not_declared(node, name + ".omega_max", rclcpp::ParameterValue(0.8));
  nav2_util::declare_parameter_if_not_declared(node, name + ".step_size", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(node, name + ".bk", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(node, name + ".scan_range_max", rclcpp::ParameterValue(5.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".max_obstacle_points", rclcpp::ParameterValue(200));
  nav2_util::declare_parameter_if_not_declared(node, name + ".desired_speed", rclcpp::ParameterValue(0.4));

  // Scan topics — direct sensor input, no costmap
  nav2_util::declare_parameter_if_not_declared(node, name + ".scan_topics",
    rclcpp::ParameterValue(std::vector<std::string>{"/scan", "/camera/scan", "/camera_rear/scan"}));

  node->get_parameter(name + ".horizon", params_.horizon);
  node->get_parameter(name + ".dt", params_.dt);
  node->get_parameter(name + ".iterations", params_.iterations);
  node->get_parameter(name + ".q_pos", params_.q_pos);
  node->get_parameter(name + ".q_theta", params_.q_theta);
  node->get_parameter(name + ".r_v", params_.r_v);
  node->get_parameter(name + ".r_omega", params_.r_omega);
  node->get_parameter(name + ".d_safe", params_.d_safe);
  node->get_parameter(name + ".collision_weight", params_.collision_weight);
  node->get_parameter(name + ".v_max", params_.v_max);
  node->get_parameter(name + ".v_min", params_.v_min);
  node->get_parameter(name + ".omega_max", params_.omega_max);
  node->get_parameter(name + ".step_size", params_.step_size);
  node->get_parameter(name + ".bk", params_.bk);
  node->get_parameter(name + ".scan_range_max", scan_range_max_);

  int max_pts_int;
  node->get_parameter(name + ".max_obstacle_points", max_pts_int);
  max_obstacle_points_ = static_cast<size_t>(max_pts_int);

  node->get_parameter(name + ".desired_speed", desired_speed_);
  node->get_parameter(name + ".scan_topics", scan_topics_);

  optimizer_.configure(params_);

  // Load DUNE ONNX model
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".dune_model_path", rclcpp::ParameterValue(std::string("")));
  node->get_parameter(name + ".dune_model_path", dune_model_path_);
  if (!dune_model_path_.empty()) {
    if (dune_.load(dune_model_path_)) {
      RCLCPP_INFO(logger_, "DUNE neural network loaded: %s", dune_model_path_.c_str());
    } else {
      RCLCPP_WARN(logger_, "Failed to load DUNE model: %s", dune_model_path_.c_str());
    }
  }

  // Subscribe to scan topics — direct raw sensor input
  for (const auto & topic : scan_topics_) {
    auto sub = node->create_subscription<sensor_msgs::msg::LaserScan>(
      topic, rclcpp::SensorDataQoS(),
      [this, topic](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        scanCallback(msg, msg->header.frame_id);
      });
    scan_subs_.push_back(sub);
    RCLCPP_INFO(logger_, "Subscribed to scan: %s", topic.c_str());
  }

  // Create publisher for local plan visualization
  local_plan_pub_ = node->create_publisher<nav_msgs::msg::Path>("local_plan", 1);

  RCLCPP_INFO(logger_,
    "NeuPAN controller configured: horizon=%d, dt=%.2f, iterations=%d, "
    "v_max=%.2f, omega_max=%.2f, d_safe=%.2f, dune=%s, scans=%zu",
    params_.horizon, params_.dt, params_.iterations,
    params_.v_max, params_.omega_max, params_.d_safe,
    dune_.is_loaded() ? "ACTIVE" : "off",
    scan_topics_.size());
}

void NeuPANController::scanCallback(
  const sensor_msgs::msg::LaserScan::SharedPtr msg,
  const std::string & frame_id)
{
  // Store scan points in BASE_LINK frame (transform in computeVelocityCommands using known pose)
  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_->lookupTransform("base_link", frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException &) {
    return;
  }

  double tx = tf_stamped.transform.translation.x;
  double ty = tf_stamped.transform.translation.y;
  double yaw = tf2::getYaw(tf_stamped.transform.rotation);
  double cos_y = std::cos(yaw);
  double sin_y = std::sin(yaw);

  std::vector<Eigen::Vector2d> points;
  points.reserve(msg->ranges.size());

  for (size_t i = 0; i < msg->ranges.size(); ++i) {
    float r = msg->ranges[i];
    if (r < msg->range_min || r > std::min(msg->range_max, static_cast<float>(scan_range_max_))
        || !std::isfinite(r)) {
      continue;
    }

    double angle = msg->angle_min + i * msg->angle_increment;
    double sx = r * std::cos(angle);
    double sy = r * std::sin(angle);

    // Transform to base_link frame
    double bx = cos_y * sx - sin_y * sy + tx;
    double by = sin_y * sx + cos_y * sy + ty;

    points.emplace_back(bx, by);
  }

  // Downsample if too many
  if (points.size() > max_obstacle_points_) {
    size_t step = points.size() / max_obstacle_points_;
    std::vector<Eigen::Vector2d> downsampled;
    downsampled.reserve(max_obstacle_points_);
    for (size_t i = 0; i < points.size(); i += step) {
      downsampled.push_back(points[i]);
    }
    points = std::move(downsampled);
  }

  std::lock_guard<std::mutex> lock(scan_mutex_);
  latest_scan_points_ = std::move(points);  // base_link frame
}

std::vector<Eigen::Vector2d> NeuPANController::getObstaclePoints()
{
  std::lock_guard<std::mutex> lock(scan_mutex_);
  return latest_scan_points_;  // base_link frame
}

void NeuPANController::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning up NeuPAN controller");
  scan_subs_.clear();
  local_plan_pub_.reset();
}

void NeuPANController::activate()
{
  RCLCPP_INFO(logger_, "Activating NeuPAN controller");
  has_warm_start_ = false;
}

void NeuPANController::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating NeuPAN controller");
}

void NeuPANController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
  has_warm_start_ = false;
}

geometry_msgs::msg::TwistStamped NeuPANController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  auto transformed_plan = transformGlobalPlan(pose);

  double yaw = tf2::getYaw(pose.pose.orientation);
  State x0(pose.pose.position.x, pose.pose.position.y, yaw);

  NRMPParams current_params = params_;
  double effective_speed = desired_speed_;
  if (speed_limit_ > 0.0) {
    if (speed_limit_is_percentage_) {
      current_params.v_max = params_.v_max * speed_limit_ / 100.0;
    } else {
      current_params.v_max = std::min(params_.v_max, speed_limit_);
    }
    effective_speed = std::min(effective_speed, current_params.v_max);
    optimizer_.configure(current_params);
  }

  Eigen::MatrixXd reference = interpolate_reference(
    x0, transformed_plan, params_.horizon, params_.dt, effective_speed);

  // Get obstacles from costmap (reliable) + scan points (if available)
  auto costmap = costmap_ros_->getCostmap();
  auto obstacles = ObstacleExtractor::extract(
    *costmap, x0(0), x0(1), 3.0, 253, 200);

  // Also merge raw scan points if available
  auto scan_points_base = getObstaclePoints();
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  for (const auto & p : scan_points_base) {
    double wx = cos_yaw * p(0) - sin_yaw * p(1) + x0(0);
    double wy = sin_yaw * p(0) + cos_yaw * p(1) + x0(1);
    obstacles.emplace_back(wx, wy);
  }

  // Prepare initial controls: warm start > DUNE > proportional
  const Eigen::MatrixXd * warm = nullptr;
  Eigen::MatrixXd init_controls;
  if (has_warm_start_ && prev_controls_.cols() == params_.horizon) {
    init_controls.resize(2, params_.horizon);
    init_controls.leftCols(params_.horizon - 1) =
      prev_controls_.rightCols(params_.horizon - 1);
    init_controls.col(params_.horizon - 1).setZero();
    warm = &init_controls;
  } else {
    // DUNE uses scan_points_base (robot frame) directly — no costmap!
    init_controls = dune_.generate_initial_controls(
      x0, obstacles, reference, params_.horizon, params_.dt, current_params.v_max);
    warm = &init_controls;
  }

  auto result = optimizer_.solve(x0, reference, obstacles, warm);

  prev_controls_ = result.controls;
  has_warm_start_ = true;

  if (speed_limit_ > 0.0) {
    optimizer_.configure(params_);
  }

  publishLocalPlan(result.trajectory);

  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = clock_->now();
  cmd.header.frame_id = pose.header.frame_id;
  cmd.twist.linear.x = result.controls(0, 0);
  cmd.twist.angular.z = result.controls(1, 0);

  return cmd;
}

void NeuPANController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  speed_limit_ = speed_limit;
  speed_limit_is_percentage_ = percentage;
}

nav_msgs::msg::Path NeuPANController::transformGlobalPlan(
  const geometry_msgs::msg::PoseStamped & pose)
{
  nav_msgs::msg::Path transformed_plan;
  transformed_plan.header.frame_id = costmap_ros_->getGlobalFrameID();
  transformed_plan.header.stamp = pose.header.stamp;

  if (global_plan_.poses.empty()) {
    return transformed_plan;
  }

  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_->lookupTransform(
      costmap_ros_->getGlobalFrameID(),
      global_plan_.header.frame_id,
      tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(logger_, "Could not transform plan: %s", ex.what());
    return transformed_plan;
  }

  const double robot_x = pose.pose.position.x;
  const double robot_y = pose.pose.position.y;
  double min_dist_sq = std::numeric_limits<double>::max();
  size_t closest_idx = 0;

  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    geometry_msgs::msg::PoseStamped transformed_pose;
    tf2::doTransform(global_plan_.poses[i], transformed_pose, tf_stamped);

    double dx = transformed_pose.pose.position.x - robot_x;
    double dy = transformed_pose.pose.position.y - robot_y;
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      closest_idx = i;
    }

    transformed_plan.poses.push_back(transformed_pose);
  }

  size_t prune_start = (closest_idx > 2) ? closest_idx - 2 : 0;
  if (prune_start > 0) {
    transformed_plan.poses.erase(
      transformed_plan.poses.begin(),
      transformed_plan.poses.begin() + static_cast<long>(prune_start));
  }

  return transformed_plan;
}

void NeuPANController::publishLocalPlan(const Eigen::MatrixXd & trajectory)
{
  nav_msgs::msg::Path local_plan;
  local_plan.header.stamp = clock_->now();
  local_plan.header.frame_id = costmap_ros_->getGlobalFrameID();

  for (int k = 0; k < trajectory.cols(); ++k) {
    geometry_msgs::msg::PoseStamped p;
    p.header = local_plan.header;
    p.pose.position.x = trajectory(0, k);
    p.pose.position.y = trajectory(1, k);
    p.pose.position.z = 0.0;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, trajectory(2, k));
    p.pose.orientation.x = q.x();
    p.pose.orientation.y = q.y();
    p.pose.orientation.z = q.z();
    p.pose.orientation.w = q.w();
    local_plan.poses.push_back(p);
  }

  local_plan_pub_->publish(local_plan);
}

}  // namespace neupan_controller

PLUGINLIB_EXPORT_CLASS(neupan_controller::NeuPANController, nav2_core::Controller)
