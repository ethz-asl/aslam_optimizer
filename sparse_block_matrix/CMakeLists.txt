cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

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


# For building the CHOLMOD / CSPARSE solvers 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)
FIND_PACKAGE(SuiteSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})

#generate flag export files based on the values found by FIND_PACKAGE.
#They are incorporated by a backticks expression in the manifest.xml's export/cpp tag.
file(WRITE build/cflags.export "-I${CSPARSE_INCLUDE_DIR}\n-I${CHOLMOD_INCLUDE_DIR}\n")
file(WRITE build/lflags.export "")
foreach(lib ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARIES})
  string(SUBSTRING ${lib} 0 1 lstart)
  string(COMPARE EQUAL ${lstart} "-" isflag)
  if( isflag )
	file(APPEND build/lflags.export "${lib}\n")
  else()
    file(APPEND build/lflags.export "-l${lib}\n")
  endif()
endforeach()
 
IF(NOT CSPARSE_FOUND)
   message( FATAL_ERROR "CSparse was not found")
ENDIF()

IF(NOT CHOLMOD_FOUND)
   message( FATAL_ERROR "Cholmod was not found")
ENDIF()

#common commands for building c++ executables and libraries
rosbuild_add_library(${PROJECT_NAME} 
  src/matrix_structure.cpp
  src/sparse_helper.cpp
  src/csparse_helper.cpp
  src/marginal_covariance_cholesky.cpp
  )

target_link_libraries(${PROJECT_NAME} ${CHOLMOD_LIBRARIES} ${CSPARSE_LIBRARY} )

rosbuild_add_gtest( ${PROJECT_NAME}_tests 
  test/test_main.cpp
  test/solver_tests.cpp
  test/sparse_block_matrix_tests.cpp
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})


## PYTHON EXPORTS.
# Use this handy cmake function
rosbuild_find_ros_package(numpy_eigen)
include(${numpy_eigen_PACKAGE_PATH}/cmake/add_python_export_library.cmake)

# This functions take TARGET_NAME PYTHON_MODULE_DIRECTORY sourceFile1 [sourceFile2 ...]
add_python_export_library(${PROJECT_NAME}_python src/${PROJECT_NAME}
  src/python/module.cpp
)

target_link_libraries(${PROJECT_NAME}_python ${PROJECT_NAME})
