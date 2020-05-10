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
target_link_libraries(${PROJECT_NAME}
	/usr/lib/x86_64-linux-gnu/libamd.so
/usr/lib/x86_64-linux-gnu/libbtf.so
/usr/lib/x86_64-linux-gnu/libcamd.so
/usr/lib/x86_64-linux-gnu/libccolamd.so
/usr/lib/x86_64-linux-gnu/libcholmod.so
/usr/lib/x86_64-linux-gnu/libcolamd.so
/usr/lib/x86_64-linux-gnu/libcxsparse.so
/usr/lib/x86_64-linux-gnu/libgraphblas.so
/usr/lib/x86_64-linux-gnu/libklu.so
/usr/lib/x86_64-linux-gnu/libldl.so
/usr/lib/x86_64-linux-gnu/librbio.so
/usr/lib/x86_64-linux-gnu/libspqr.so
/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
/usr/lib/x86_64-linux-gnu/libumfpack.so
)
include_directories(/usr/include/suitesparse)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest( ${PROJECT_NAME}_tests
    test/test_main.cpp
    test/solver_tests.cpp
    test/sparse_block_matrix_tests.cpp
  )
  if(TARGET ${PROJECT_NAME}_tests)
    target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} ${TBB_LIBRARIES})
  endif()
endif()

cs_install()
cs_export()
