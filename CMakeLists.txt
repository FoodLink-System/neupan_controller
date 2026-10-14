cmake_minimum_required(VERSION 3.8)
project(neupan_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(dependencies
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  tf2_ros
  tf2_geometry_msgs
)

# ── Core library (no ROS node deps, testable in isolation) ──
add_library(neupan_core SHARED
  src/diff_drive_model.cpp
  src/nrmp_optimizer.cpp
  src/obstacle_extractor.cpp
)
target_include_directories(neupan_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neupan_core Eigen3::Eigen)
ament_target_dependencies(neupan_core nav2_costmap_2d rclcpp)

# ── OSQP (QP solver for NRMP optimizer) ──
find_package(osqp QUIET)
find_package(OsqpEigen QUIET)
if(osqp_FOUND AND OsqpEigen_FOUND)
  target_link_libraries(neupan_core OsqpEigen::OsqpEigen)
  target_compile_definitions(neupan_core PUBLIC NEUPAN_HAS_OSQP=1)
  message(STATUS "NeuPAN: OSQP + OsqpEigen found — proper QP solver enabled")
else()
  message(STATUS "NeuPAN: OSQP NOT found — using gradient descent fallback")
endif()

# ── ONNX Runtime (optional — for DUNE neural network) ──
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
  find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib)
  find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATHS /usr/local/include)
endif()

# ── Plugin shared library ──
add_library(${PROJECT_NAME} SHARED
  src/neupan_controller.cpp
  src/dune_inference.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} neupan_core Eigen3::Eigen)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

if(onnxruntime_FOUND)
  target_link_libraries(${PROJECT_NAME} onnxruntime::onnxruntime)
  target_compile_definitions(${PROJECT_NAME} PRIVATE NEUPAN_HAS_ONNX=1)
  message(STATUS "NeuPAN: ONNX Runtime found — DUNE neural network enabled")
elseif(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
  target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
  target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE})
  target_compile_definitions(${PROJECT_NAME} PRIVATE NEUPAN_HAS_ONNX=1)
  message(STATUS "NeuPAN: ONNX Runtime found (manual) — DUNE neural network enabled")
else()
  message(STATUS "NeuPAN: ONNX Runtime NOT found — DUNE will use analytical fallback")
endif()

pluginlib_export_plugin_description_file(nav2_core neupan_controller_plugin.xml)

# ── Install ──
install(TARGETS neupan_core ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES neupan_controller_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ──
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_diff_drive_model test/test_diff_drive_model.cpp)
  target_link_libraries(test_diff_drive_model neupan_core Eigen3::Eigen)
  target_include_directories(test_diff_drive_model PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  ament_add_gtest(test_nrmp_optimizer test/test_nrmp_optimizer.cpp)
  target_link_libraries(test_nrmp_optimizer neupan_core Eigen3::Eigen)
  target_include_directories(test_nrmp_optimizer PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  ament_add_gtest(test_obstacle_extractor test/test_obstacle_extractor.cpp)
  target_link_libraries(test_obstacle_extractor neupan_core Eigen3::Eigen)
  ament_target_dependencies(test_obstacle_extractor nav2_costmap_2d rclcpp)
  target_include_directories(test_obstacle_extractor PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})

ament_package()
