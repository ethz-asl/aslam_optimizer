/*
 * LeapFrog.hpp
 *
 *  Created on: 04.08.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#ifndef INCLUDE_ASLAM_BACKEND_LEAPFROG_HPP_
#define INCLUDE_ASLAM_BACKEND_LEAPFROG_HPP_

#include <aslam/backend/util/CommonDefinitions.hpp>
#include <aslam/backend/util/CostFunctionInterface.hpp>

namespace aslam {
namespace backend {
namespace leapfrog {

/**
 * Leap-Frog simulation
 * @param potEnergy Holding the potential energy function
 * @param gradient Gradient of the potential energy function at the current system state
 * @param momentum Initial momentum
 * @param numSteps Number of leap-frog steps for simulation
 * @param stepLength Step length for each step
 * @return True iff simulation seems OK at the end, false if simulation has diverged
 */
bool simulate(const boost::shared_ptr<CostFunctionInterface>& potEnergy,
              RowVectorType& gradient,
              ColumnVectorType& momentum,
              const std::size_t numSteps,
              const double stepLength);


} /* namespace leapfrog */
} /* namespace backend */
} /* namespace aslam */

#endif /* INCLUDE_ASLAM_BACKEND_LEAPFROG_HPP_ */
