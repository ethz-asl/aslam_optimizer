cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)
#set(ROS_BUILD_TYPE Release)
# just write the upper line into LocalBuildType.cmake to have a ROS_BUILD_TYPE not under version control.
include(LocalBuildType.cmake OPTIONAL)
message("ROS_BUILD_TYPE=" ${ROS_BUILD_TYPE})

rosbuild_init()

#add_definitions (-Wall -std=c++0x -O3 -march=native)

set(CMAKE_CXX_FLAGS "-std=c++0x")

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


rosbuild_add_library(${PROJECT_NAME}
  src/RotationExpression.cpp 
  src/RotationExpressionNode.cpp
  src/RotationQuaternion.cpp
  src/MappedRotationQuaternion.cpp
  src/Vector2RotationQuaternionExpressionAdapter.cpp

  src/MatrixExpression.cpp 
  src/MatrixExpressionNode.cpp
  src/MatrixTransformation.cpp

  src/TransformationExpression.cpp 
  src/TransformationExpressionNode.cpp
  src/TransformationBasic.cpp

  src/EuclideanExpression.cpp 
  src/EuclideanExpressionNode.cpp
  src/EuclideanPoint.cpp
  src/MappedEuclideanPoint.cpp

  src/HomogeneousExpression.cpp 
  src/HomogeneousExpressionNode.cpp
  src/HomogeneousPoint.cpp
  src/MappedHomogeneousPoint.cpp

  src/ScalarExpression.cpp
  src/ScalarExpressionNode.cpp
  src/Scalar.cpp

  src/EuclideanDirection.cpp

  src/ErrorTermTransformation.cpp

  src/MapTransformation.cpp
  )


rosbuild_add_gtest(${PROJECT_NAME}_test
  test/test_main.cpp
  test/RotationExpression.cpp
  test/HomogeneousExpression.cpp
  test/MatrixAndEuclideanExpression.cpp
  test/GenericMatrixExpression.cpp
  test/QuaternionExpression.cpp
  #test/CameraDesignVariable.cpp
  test/ScalarExpression.cpp
  test/ErrorTest_Transformation.cpp
  )

target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
