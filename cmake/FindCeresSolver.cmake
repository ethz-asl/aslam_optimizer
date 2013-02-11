# - Finds Ceres Solver support
# http://code.google.com/p/ceres-solver/
#
# I tried to make things easy by including requirements glog, gflags, protobuf, suite sparse, and openmp.
#
# The following variables are set:
#   CeresSolver_CXX_FLAGS    - flags to add to the CXX compiler for OpenMP support
#   CeresSolver_INCLUDE_DIRS - include directories
#   CeresSolver_LIBRERIES    - libraries to link against
#   CeresSolver_FOUND        - true if openmp is detected
#

FIND_PATH(CeresSolver_INCLUDE_DIR NAMES ceres.h
    PATHS
    /usr/include/ceres
    /usr/local/include/ceres
    NO_DEFAULT_PATH
  )

FIND_LIBRARY(CeresSolver_LIBRARY NAMES ceres
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  NO_DEFAULT_PATH
  )

FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(SuiteSparse REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Gflags REQUIRED)
FIND_PACKAGE(Protobuf REQUIRED)



SET(CeresSolver_FOUND FALSE)
IF(CeresSolver_INCLUDE_DIR AND CeresSolver_LIBRARY)
  # strip off the ceres part of the path so that includes look like <ceres/ceres.h>
  get_filename_component(CeresSolver_INCLUDE_DIR ${CeresSolver_INCLUDE_DIR} PATH)

  SET(CeresSolver_LIBRARIES 
    ${CeresSolver_LIBRARY}   
    ${CSPARSE_LIBRARY} 
    ${CHOLMOD_LIBRARIES} 
    ${GFlags_LIBS}
    ${GLOG_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    )
  SET(CeresSolver_INCLUDE_DIRS
    ${CeresSolver_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR} 
    ${GFlags_INCLUDE_DIRS} 
    ${CHOLMOD_INCLUDE_DIR} 
    ${PROTOBUF_INCLUDE_DIRS}
    ${GFlags_INCLUDE_DIRS}   
    )

  SET(CeresSolver_CXX_FLAGS ${OpenMP_CXX_FLAGS})

  SET(CeresSolver_FOUND TRUE)
  
  
ENDIF(CeresSolver_INCLUDE_DIR AND CeresSolver_LIBRARY)

