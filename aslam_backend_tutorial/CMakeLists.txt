cmake_minimum_required(VERSION 2.8.3)
project(aslam_backend_tutorial)

find_package(catkin_simple REQUIRED)
catkin_simple()

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
target_link_libraries(example ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_test
  test/test_main.cpp
  test/ErrorTests.cpp
)
# Tell CMake to link against the project.
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
