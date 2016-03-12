/*
 * utils.hpp
 *
 *  Created on: 24.09.2015
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_UTIL_UTILS_HPP_
#define INCLUDE_ASLAM_BACKEND_UTIL_UTILS_HPP_

#include <aslam/backend/JacobianContainer.hpp>

namespace aslam {
namespace backend {
namespace utils {

/** branchless signum method */
template <typename T>
inline int sign(const T& val) {
  return (0.0 < val) - (val < 0.0);
}

/** power-2 */
template <typename T>
inline T sqr(const T& val) {
  return val*val;
}

}
}
}

#endif /* INCLUDE_ASLAM_BACKEND_UTIL_UTILS_HPP_ */
