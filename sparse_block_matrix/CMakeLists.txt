cmake_minimum_required(VERSION 2.8.3)

project(sparse_block_matrix)
find_package(catkin_simple REQUIRED)
catkin_simple()

cs_add_library(${PROJECT_NAME} 
  src/matrix_structure.cpp
  src/sparse_helper.cpp
  src/csparse_helper.cpp
  src/marginal_covariance_cholesky.cpp
)

catkin_add_gtest( ${PROJECT_NAME}_tests 
  test/test_main.cpp
  test/solver_tests.cpp
  test/sparse_block_matrix_tests.cpp
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

cs_install()
cs_export()


