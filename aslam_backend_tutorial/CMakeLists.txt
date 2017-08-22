cmake_minimum_required(VERSION 2.8.3)
project(aslam_backend_tutorial)

find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(Boost REQUIRED COMPONENTS system) 
include_directories(${Boost_INCLUDE_DIRS})

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # deprecated-register warnings are cause by old Eigen versions
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()

# enable C++11 support
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

cs_add_library(${PROJECT_NAME}
  src/ScalarDesignVariable.cpp
  src/ErrorTermObservation.cpp
  src/ErrorTermMotion.cpp
  src/ErrorTermPrior.cpp
)

# This function takes TARGET_NAME PYTHON_MODULE_DIRECTORY sourceFile1 [sourceFile2 ...]
add_python_export_library(${PROJECT_NAME}_python python/${PROJECT_NAME}
  src/python_module.cpp
)
# Tell CMake to link against the project.
target_link_libraries(${PROJECT_NAME}_python ${PROJECT_NAME})


# Build the c++ example
cs_add_executable( example
  src/example.cpp
)
# Tell CMake to link against the project.
target_link_libraries(example ${PROJECT_NAME} ${Boost_LIBRARIES})

catkin_add_gtest(${PROJECT_NAME}_test
  test/test_main.cpp
  test/ErrorTests.cpp
)
# Tell CMake to link against the project.
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
