cmake_minimum_required(VERSION 2.8.3)

project(sparse_block_matrix)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-register") # deprecated-register warnings are cause by old Eigen versions

# enable C++11 support
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

cs_add_library(${PROJECT_NAME} 
  src/matrix_structure.cpp
  src/sparse_helper.cpp
  src/marginal_covariance_cholesky.cpp
)

catkin_add_gtest( ${PROJECT_NAME}_tests 
  test/test_main.cpp
  test/solver_tests.cpp
  test/sparse_block_matrix_tests.cpp
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} ${TBB_LIBRARIES})

cs_install()
cs_export()
