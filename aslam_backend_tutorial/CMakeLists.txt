cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set(CMAKE_CXX_FLAGS "-std=c++0x")

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
  src/ScalarDesignVariable.cpp
  src/ErrorTermObservation.cpp
  src/ErrorTermMotion.cpp
  src/ErrorTermPrior.cpp
)


rosbuild_find_ros_package(numpy_eigen)
include(${numpy_eigen_PACKAGE_PATH}/cmake/add_python_export_library.cmake)

# This function takes TARGET_NAME PYTHON_MODULE_DIRECTORY sourceFile1 [sourceFile2 ...]
add_python_export_library(${PROJECT_NAME}_python python/${PROJECT_NAME}
  src/python_module.cpp
)

# Tell CMake to link against the project.
target_link_libraries(${PROJECT_NAME}_python ${PROJECT_NAME})


# Build the c++ example
rosbuild_add_executable( example
  src/example.cpp
)

# Tell CMake to link against the project.
target_link_libraries(example ${PROJECT_NAME})

# Avoid clash with tr1::tuple: https://code.google.com/p/googletest/source/browse/trunk/README?r=589#257
add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)

rosbuild_add_gtest(${PROJECT_NAME}_test
  test/test_main.cpp
  test/ErrorTests.cpp
)
# Tell CMake to link against the project.
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
