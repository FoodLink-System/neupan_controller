#include <gtest/gtest.h>
#include "neupan_controller/obstacle_extractor.hpp"

using namespace neupan_controller;

TEST(ObstacleExtractor, EmptyCostmap)
{
  nav2_costmap_2d::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  auto obs = ObstacleExtractor::extract(costmap, 0.25, 0.25, 1.0);
  EXPECT_TRUE(obs.empty());
}

TEST(ObstacleExtractor, SingleObstacle)
{
  nav2_costmap_2d::Costmap2D costmap(20, 20, 0.05, 0.0, 0.0);
  costmap.setCost(10, 10, 254);
  auto obs = ObstacleExtractor::extract(costmap, 0.5, 0.5, 1.0, 253);
  ASSERT_EQ(obs.size(), 1u);
  EXPECT_NEAR(obs[0](0), 0.525, 0.01);
  EXPECT_NEAR(obs[0](1), 0.525, 0.01);
}

TEST(ObstacleExtractor, RadiusFilter)
{
  nav2_costmap_2d::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
  costmap.setCost(5, 5, 254);
  costmap.setCost(90, 90, 254);
  auto obs = ObstacleExtractor::extract(costmap, 0.25, 0.25, 1.0, 253);
  ASSERT_EQ(obs.size(), 1u);
}

TEST(ObstacleExtractor, MaxPointsCap)
{
  nav2_costmap_2d::Costmap2D costmap(20, 20, 0.05, 0.0, 0.0);
  for (unsigned int i = 0; i < 20; ++i) {
    costmap.setCost(i, 10, 254);
  }
  auto obs = ObstacleExtractor::extract(costmap, 0.5, 0.5, 2.0, 253, 5);
  EXPECT_LE(obs.size(), 5u);
}
