#ifndef INCLUDE_ASLAM_BACKEND_COMMON_HPP_
#define INCLUDE_ASLAM_BACKEND_COMMON_HPP_

#include <Eigen/Core>

#include <sm/timing/Timer.hpp>

namespace aslam {
namespace backend {

#ifdef aslam_backend_ENABLE_TIMING
  typedef sm::timing::Timer Timer;
#else
  typedef sm::timing::DummyTimer Timer;
#endif
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
  typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorType;

}
}

#endif /* INCLUDE_ASLAM_BACKEND_COMMON_HPP_ */
