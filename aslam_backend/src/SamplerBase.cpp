/*
 * SamplerBase.cpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#include <aslam/backend/SamplerBase.hpp>

namespace aslam {
namespace backend {


SamplerBase::Statistics::Statistics() :
  nIterationsThisRun(0),
  nSamplesAcceptedThisRun(0),
  nIterationsTotal(0),
  nSamplesAcceptedTotal(0) {

}

void SamplerBase::Statistics::reset() {
  nIterationsThisRun = 0;
  nIterationsTotal = 0;
  nSamplesAcceptedThisRun = 0;
  nSamplesAcceptedTotal = 0;
}

/// \brief Getter for the acceptance rate
double SamplerBase::Statistics::getAcceptanceRate(bool total /*= false*/) const {
  return getNumIterations(total) > 0 ?
      static_cast<double>(getNumAcceptedSamples(total))/static_cast<double>(getNumIterations(total)) : 0.0;
}

/// \brief Getter for the number of iterations since the last run() or initialize() call
std::size_t SamplerBase::Statistics::getNumIterations(bool total /*= false*/) const {
  return total ? nIterationsTotal : nIterationsThisRun;
}

/// \brief Getter for the number of iterations since the last run() or initialize() call
std::size_t SamplerBase::Statistics::getNumAcceptedSamples(bool total /*= false*/) const {
  return total ? nSamplesAcceptedTotal : nSamplesAcceptedThisRun;
}



/// \brief Run the sampler for at maximum \p nStepsMax until \p nAcceptedSamples samples were accepted
void SamplerBase::run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples /*= std::numeric_limits<std::size_t>::max()*/) {

  SM_ASSERT_GT(Exception, nStepsMax, 0, "It does not make sense to run the sampler with no steps.");
  SM_ASSERT_GT(Exception, nAcceptedSamples, 0, "It does not make sense to run the sampler until zero samples were accepted.");

  if (!isInitialized())
    initialize();

  _statistics.nSamplesAcceptedThisRun = 0;
  _statistics.nIterationsThisRun = 0;

  runImplementation(nStepsMax, nAcceptedSamples, _statistics);

  _statistics.nSamplesAcceptedTotal += _statistics.nSamplesAcceptedThisRun;
  _statistics.nIterationsTotal += _statistics.nIterationsThisRun;

}

/// \brief Set up to work on the log density. The log density may neglect the normalization constant.
void SamplerBase::setNegativeLogDensity(boost::shared_ptr<OptimizationProblemBase> negLogDensity) {
  setProblem(negLogDensity);
}

/// \brief Mutable getter for the log density formulation
boost::shared_ptr<OptimizationProblemBase> SamplerBase::getNegativeLogDensity() {
  return getProblem();
}

/// \brief Const getter for the log density formulation
boost::shared_ptr<const OptimizationProblemBase> SamplerBase::getNegativeLogDensity() const {
  return getProblem();
}

/// \brief Signal the sampler that the negative log density formulation changed.
void SamplerBase::signalNegativeLogDensityChanged() {
  setInitialized(false);
}

/// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
///        hooked up to design variables and running finite differences on error terms where this is possible.
void SamplerBase::checkNegativeLogDensitySetup() const {
  checkProblemSetup();
}

/// \brief Evaluate the current negative log density
double SamplerBase::evaluateNegativeLogDensity() const {
  return evaluateError();
}

/// \brief Initialization method
void SamplerBase::initialize() {
  _statistics.reset();
  ScalarOptimizerBase::initialize();
}

/// \brief Const getter for statistics
const SamplerBase::Statistics& SamplerBase::statistics() const {
  return _statistics;
}

}
}
