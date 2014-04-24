cmake_minimum_required(VERSION 2.8.3)
project(aslam_backend_python)

find_package(catkin_simple REQUIRED)
catkin_simple()

include_directories(${EIGEN_INCLUDE_DIRS})

IF(${CMAKE_SYSTEM} MATCHES "Darwin-*")
  message("You are using OSX")
  add_definitions(  -fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long)
ELSE()
  add_definitions( -fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++0x)

ENDIF(${CMAKE_SYSTEM} MATCHES "Darwin-*")
# This functions take TARGET_NAME PYTHON_MODULE_DIRECTORY sourceFile1 [sourceFile2 ...]
add_python_export_library(${PROJECT_NAME} python/aslam_backend
  src/module.cpp
  src/Backend.cpp
  src/Optimizer.cpp
  src/OptimizerOptions.cpp
  src/OptimizationProblem.cpp
  src/DesignVariable.cpp
  src/ErrorTerm.cpp
  src/MEstimators.cpp
  src/JacobianContainer.cpp
  src/BackendExpressions.cpp
  src/CompressedColumnMatrix.cpp
  src/LinearSystemSolver.cpp
  src/ErrorTermTransformation.cpp
  src/TrustRegionPolicies.cpp
)


cs_install()
cs_export()



