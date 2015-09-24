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

inline bool isFinite(const JacobianContainer& jc, const DesignVariable& dv) {
  return jc.Jacobian(&dv).allFinite();
}

}
}
}

#endif /* INCLUDE_ASLAM_BACKEND_UTIL_UTILS_HPP_ */
