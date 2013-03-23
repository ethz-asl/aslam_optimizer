cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE Debug)
set(ROS_BUILD_TYPE Release)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

IF(${CMAKE_SYSTEM} MATCHES "Darwin-*")
  message("You are using OSX")
  #SET (CMAKE_C_COMPILER         "/usr/bin/gcc")
  #SET (CMAKE_CXX_COMPILER        "/usr/bin/g++")
  #SET (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
  add_definitions( -std=c++0x -Wno-unused-local-typedefs -fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long)
ELSE()
  add_definitions( -fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++0x)
ENDIF(${CMAKE_SYSTEM} MATCHES "Darwin-*")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)
FIND_PACKAGE(TBB)
SET(EXTRA_LIBS)
IF(TBB_FOUND)
  SET(EXTRA_LIBS ${TBB_LIBRARY})
ENDIF(TBB_FOUND)

rosbuild_add_library(${PROJECT_NAME} 
  src/MEstimatorPolicies.cpp
  src/JacobianContainer.cpp
  src/DesignVariable.cpp
  src/ErrorTerm.cpp
  src/OptimizationProblemBase.cpp
  src/Optimizer.cpp
  src/sparse_matrix_functions.cpp
  src/OptimizationProblem.cpp
  src/JacobianBuilder.cpp
  # src/CompressedColumnJacobianTransposeBuilder.cpp
  src/LinearSystemSolver.cpp
  src/BlockCholeskyLinearSystemSolver.cpp
  src/Optimizer2.cpp
 # src/Cholmod.cpp
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
  )

target_link_libraries(${PROJECT_NAME} ${CHOLMOD_LIBRARIES} ${EXTRA_LIBS}  ${CSPARSE_LIBRARY})
rosbuild_link_boost(${PROJECT_NAME} thread system)
rosbuild_add_gtest(${PROJECT_NAME}_test
  test/test_main.cpp
  test/JacobianContainer.cpp
  test/test_sparse_matrix_functions.cpp
  test/TestOptimizer.cpp
  test/TestOptimizationProblem.cpp
  test/CompressedColumnMatrixTest.cpp
  test/MatrixTestHarness.cpp
  test/DenseMatrixTest.cpp
  test/SparseMatrixTest.cpp
  test/LinearSolverTests.cpp
  test/ErrorTermTests.cpp
  )

target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})

