cmake_minimum_required(VERSION 2.8.3)
project(aslam_backend)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

find_package(SUITESPARSE REQUIRED)
include_directories(${SUITESPARSE_INCLUDE_DIRS})

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # deprecated-register warnings are cause by old Eigen versions
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()

# enable C++11 support
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

# enable timing via cmake option -Daslam_backend_ENABLE_TIMING
SET(aslam_backend_ENABLE_TIMING OFF CACHE BOOL "Enable sm::timing based profiling in aslam_backend")
# cmake -Daslam_backend_ENABLE_TIMING=1 or catkin build -Daslam_backend_ENABLE_TIMING
if (${PROJECT_NAME}_ENABLE_TIMING)
  message(STATUS "${PROJECT_NAME}: Timing enabled")
  add_definitions(-D${PROJECT_NAME}_ENABLE_TIMING)
endif()

cs_add_library(${PROJECT_NAME}
  src/MEstimatorPolicies.cpp
  src/JacobianContainerSparse.cpp
  src/JacobianContainerDense.cpp
  src/DesignVariable.cpp
  src/ErrorTerm.cpp
  src/ScalarNonSquaredErrorTerm.cpp
  src/OptimizationProblemBase.cpp
  src/LineSearch.cpp
  src/Optimizer.cpp
  src/sparse_matrix_functions.cpp
  src/OptimizationProblem.cpp
  src/SimpleOptimizationProblem.cpp
  src/JacobianBuilder.cpp
  src/LinearSystemSolver.cpp
  src/BlockCholeskyLinearSystemSolver.cpp
  src/SparseCholeskyLinearSystemSolver.cpp
  src/SparseQrLinearSystemSolver.cpp
  src/Matrix.cpp
  src/DenseMatrix.cpp
  src/SparseBlockMatrixWrapper.cpp
  src/DenseQrLinearSystemSolver.cpp
  src/BlockCholeskyLinearSolverOptions.cpp
  src/SparseCholeskyLinearSolverOptions.cpp
  src/SparseQRLinearSolverOptions.cpp
  src/DenseQRLinearSolverOptions.cpp
  src/TrustRegionPolicy.cpp
  src/ErrorTermDs.cpp
  src/GaussNewtonTrustRegionPolicy.cpp
  src/LevenbergMarquardtTrustRegionPolicy.cpp
  src/Marginalizer.cpp
  src/MarginalizationPriorErrorTerm.cpp
  src/DogLegTrustRegionPolicy.cpp
  src/SamplerBase.cpp
  src/OptimizerBase.cpp
  src/Optimizer2.cpp
  src/OptimizerRprop.cpp
  src/OptimizerBFGS.cpp
  src/ProbDataAssocPolicy.cpp
  src/SamplerMetropolisHastings.cpp
  src/SamplerHybridMcmc.cpp
  src/util/ThreadedRangeProcessor.cpp
  src/util/ProblemManager.cpp
  src/OptimizerCallbackManager.cpp
  src/LineSearchTrustRegionPolicy.cpp
)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${SUITESPARSE_LIBRARY_DIRS})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(${PROJECT_NAME}_test
    test/test_main.cpp
    test/JacobianContainer.cpp
    test/test_sparse_matrix_functions.cpp
    test/TestProblemManager.cpp
    test/TestLineSearch.cpp
    test/TestOptimizerBase.cpp
    test/TestOptimizer.cpp
    test/TestOptimizer2.cpp
    test/TestOptimizerRprop.cpp
    test/TestOptimizerBFGS.cpp
    test/TestSamplerMcmc.cpp
    test/CallbackTest.cpp
    test/TestOptimizationProblem.cpp
    test/CompressedColumnMatrixTest.cpp
    test/MatrixTestHarness.cpp
    test/DenseMatrixTest.cpp
    test/SparseMatrixTest.cpp
    test/LinearSolverTests.cpp
    test/ErrorTermTests.cpp
    test/ProbDataAssocPolicyTest.cpp
    test/MatrixStackTest.cpp
  )
  if(TARGET ${PROJECT_NAME}_test)
    target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
  endif()
endif()

cs_install()
cs_export()
