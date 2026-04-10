#include "neupan_controller/dune_inference.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef NEUPAN_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace neupan_controller
{

struct DUNEInference::Impl
{
#ifdef NEUPAN_HAS_ONNX
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dune"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int n_constraints{0};
#endif
};

DUNEInference::DUNEInference() : impl_(std::make_unique<Impl>()) {}
DUNEInference::~DUNEInference() = default;
DUNEInference::DUNEInference(DUNEInference &&) noexcept = default;
DUNEInference & DUNEInference::operator=(DUNEInference &&) noexcept = default;

bool DUNEInference::load(const std::string & model_path)
{
#ifdef NEUPAN_HAS_ONNX
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), opts);

    auto output_info = impl_->session->GetOutputTypeInfo(0);
    auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    impl_->n_constraints = (shape.size() > 1) ? static_cast<int>(shape[1]) : 4;

    loaded_ = true;
    std::cout << "DUNE model loaded: " << model_path
              << " (output dim: " << impl_->n_constraints << ")" << std::endl;
    return true;
  } catch (const Ort::Exception & e) {
    std::cerr << "Failed to load DUNE model: " << e.what() << std::endl;
    loaded_ = false;
    return false;
  }
#else
  (void)model_path;
  std::cout << "DUNE: ONNX Runtime not available, using analytical fallback" << std::endl;
  return false;
#endif
}

Eigen::MatrixXd DUNEInference::predict(const std::vector<Eigen::Vector2d> & obstacle_points)
{
#ifdef NEUPAN_HAS_ONNX
  if (!loaded_ || obstacle_points.empty() || !impl_->session) {
    return Eigen::MatrixXd::Zero(0, impl_->n_constraints);
  }

  const size_t n = obstacle_points.size();
  std::vector<float> input_data(n * 2);
  for (size_t i = 0; i < n; ++i) {
    input_data[i * 2 + 0] = static_cast<float>(obstacle_points[i](0));
    input_data[i * 2 + 1] = static_cast<float>(obstacle_points[i](1));
  }

  std::array<int64_t, 2> input_shape = {static_cast<int64_t>(n), 2};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    impl_->memory_info, input_data.data(), input_data.size(),
    input_shape.data(), input_shape.size());

  const char * input_names[] = {"obstacle_point"};
  const char * output_names[] = {"mu"};

  auto output_tensors = impl_->session->Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  float * output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int out_dim = (output_shape.size() > 1) ? static_cast<int>(output_shape[1]) : impl_->n_constraints;

  Eigen::MatrixXd result(n, out_dim);
  for (size_t i = 0; i < n; ++i) {
    for (int j = 0; j < out_dim; ++j) {
      result(i, j) = static_cast<double>(output_data[i * out_dim + j]);
    }
  }
  return result;
#else
  (void)obstacle_points;
  return Eigen::MatrixXd::Zero(0, 4);
#endif
}

Eigen::MatrixXd DUNEInference::generate_initial_controls(
  const Eigen::Vector3d & x0,
  const std::vector<Eigen::Vector2d> & obstacles,
  const Eigen::MatrixXd & reference,
  int horizon,
  double dt,
  double v_max)
{
  Eigen::MatrixXd controls(2, horizon);

  // Proportional controller toward reference (with optional DUNE bias)
  Eigen::Vector3d state = x0;

  // Get DUNE predictions for obstacle avoidance bias
  Eigen::MatrixXd mu;
  std::vector<Eigen::Vector2d> rel_obstacles;
  if (loaded_ && !obstacles.empty()) {
    double cos_t = std::cos(-x0(2));
    double sin_t = std::sin(-x0(2));
    rel_obstacles.reserve(obstacles.size());
    for (const auto & obs : obstacles) {
      double dx = obs(0) - x0(0);
      double dy = obs(1) - x0(1);
      rel_obstacles.emplace_back(cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy);
    }
    mu = predict(rel_obstacles);
  }

  for (int k = 0; k < horizon; ++k) {
    int ref_k = std::min(k + 1, static_cast<int>(reference.cols()) - 1);
    double dx = reference(0, ref_k) - state(0);
    double dy = reference(1, ref_k) - state(1);
    double dist = std::sqrt(dx * dx + dy * dy);
    double target_theta = std::atan2(dy, dx);
    double angle_err = target_theta - state(2);
    while (angle_err > M_PI) angle_err -= 2.0 * M_PI;
    while (angle_err < -M_PI) angle_err += 2.0 * M_PI;

    double v = std::min(v_max, std::max(0.0, dist * 2.0));
    double omega = std::clamp(angle_err * 3.0, -0.8, 0.8);

    // DUNE obstacle avoidance bias
    if (mu.rows() > 0 && !rel_obstacles.empty()) {
      double repulse_x = 0.0, repulse_y = 0.0;
      for (size_t i = 0; i < rel_obstacles.size(); ++i) {
        double mu_sum = mu.row(i).sum();
        if (mu_sum > 0.01) {
          double od = rel_obstacles[i].norm();
          if (od > 1e-3) {
            repulse_x -= mu_sum * rel_obstacles[i](0) / (od * od);
            repulse_y -= mu_sum * rel_obstacles[i](1) / (od * od);
          }
        }
      }
      double repulse_strength = std::sqrt(repulse_x * repulse_x + repulse_y * repulse_y);
      if (repulse_strength > 0.01) {
        double rep_world_x = std::cos(x0(2)) * repulse_x - std::sin(x0(2)) * repulse_y;
        double rep_world_y = std::sin(x0(2)) * repulse_x + std::cos(x0(2)) * repulse_y;
        double avoid_angle = std::atan2(rep_world_y, rep_world_x) - state(2);
        while (avoid_angle > M_PI) avoid_angle -= 2.0 * M_PI;
        while (avoid_angle < -M_PI) avoid_angle += 2.0 * M_PI;
        omega += std::clamp(avoid_angle * repulse_strength * 2.0, -0.4, 0.4);
        v *= std::max(0.2, 1.0 - repulse_strength * 0.5);
      }
    }

    if (std::abs(angle_err) > 0.3) v *= 0.3;

    controls(0, k) = std::clamp(v, 0.0, v_max);
    controls(1, k) = std::clamp(omega, -0.8, 0.8);

    state(0) += controls(0, k) * std::cos(state(2)) * dt;
    state(1) += controls(0, k) * std::sin(state(2)) * dt;
    state(2) += controls(1, k) * dt;
  }

  return controls;
}

}  // namespace neupan_controller
