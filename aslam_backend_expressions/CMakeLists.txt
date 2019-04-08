cmake_minimum_required(VERSION 2.8.3)
project(aslam_backend_expressions)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(Boost REQUIRED COMPONENTS system program_options)
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
  src/RotationExpression.cpp 
  src/RotationExpressionNode.cpp
  src/RotationQuaternion.cpp
  src/MappedRotationQuaternion.cpp
  src/Vector2RotationQuaternionExpressionAdapter.cpp

  src/MatrixExpression.cpp 
  src/MatrixExpressionNode.cpp
  src/MatrixBasic.cpp
  src/RotationScalarExpressionNode.cpp

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
  src/ErrorTermEuclidean.cpp
  src/L1Regularizer.cpp

  src/MapTransformation.cpp

  src/KinematicChain.cpp
  
  src/ExpressionNodeVisitor.cpp
  src/ToTextNodeVisitor.cpp
)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

cs_add_executable(${PROJECT_NAME}-profiling
  test/Profiling.cpp
)
target_link_libraries(${PROJECT_NAME}-profiling ${PROJECT_NAME} ${Boost_LIBRARIES})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(${PROJECT_NAME}_test
    test/test_main.cpp
    test/RotationExpression.cpp
    test/HomogeneousExpression.cpp
    test/MatrixAndEuclideanExpression.cpp
    test/ExpressionErrorTermTest.cpp
    test/FixedPointNumberTest.cpp
    test/GenericScalarExpressionTest.cpp
    test/GenericMatrixExpression.cpp
    test/QuaternionExpression.cpp
    test/CacheExpression.cpp
    test/ScalarExpression.cpp
    test/ExpressionUtils.cpp
    test/ErrorTest_Transformation.cpp
    test/ErrorTest_Euclidean.cpp
    test/ErrorTest_L1Regularizer.cpp
    test/VectorExpressionTest.cpp
    test/KinematicChain.cpp
    test/ExpressionNodeVisitorTest.cpp
  )
  if(TARGET ${PROJECT_NAME}_test)
    target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
  endif()
endif()

cs_install()
cs_export()

