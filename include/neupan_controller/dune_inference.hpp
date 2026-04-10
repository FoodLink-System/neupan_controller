#ifndef NEUPAN_CONTROLLER__DUNE_INFERENCE_HPP_
#define NEUPAN_CONTROLLER__DUNE_INFERENCE_HPP_

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

namespace neupan_controller
{

/// DUNE neural network inference via ONNX Runtime (PIMPL pattern).
class DUNEInference
{
public:
  DUNEInference();
  ~DUNEInference();
  DUNEInference(DUNEInference &&) noexcept;
  DUNEInference & operator=(DUNEInference &&) noexcept;

  bool load(const std::string & model_path);
  bool is_loaded() const { return loaded_; }

  Eigen::MatrixXd predict(const std::vector<Eigen::Vector2d> & obstacle_points);

  Eigen::MatrixXd generate_initial_controls(
    const Eigen::Vector3d & x0,
    const std::vector<Eigen::Vector2d> & obstacles,
    const Eigen::MatrixXd & reference,
    int horizon,
    double dt,
    double v_max);

private:
  bool loaded_{false};
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace neupan_controller

#endif  // NEUPAN_CONTROLLER__DUNE_INFERENCE_HPP_
