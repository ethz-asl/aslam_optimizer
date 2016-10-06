#include <gtest/gtest.h>

#include <sm/random.hpp>

/// Run all the tests that were declared with TEST()
int main(int argc, char **argv) {
//  sm::random::seed(std::time(nullptr));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

