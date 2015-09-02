/*
 * SamplerBase.cpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#include <aslam/backend/SamplerBase.hpp>

#include <sm/logging.hpp>

using namespace std;

namespace aslam {
namespace backend {


SamplerBase::Statistics::Statistics() :
  nIterations(0),
  nSamplesAcceptedTotal(0),
  nSamplesAcceptedThisRun(0),
  weightedMeanAcceptanceProbability(0.0) {

}

void SamplerBase::Statistics::reset() {
  nIterations = 0;
  nSamplesAcceptedTotal = 0;
  nSamplesAcceptedThisRun = 0;
  weightedMeanAcceptanceProbability = 0.0;
}

/// \brief Getter for the acceptance rate
double SamplerBase::Statistics::getAcceptanceRate() const {
  return getNumIterations() > 0 ? static_cast<double>(getNumAcceptedSamples(true))/static_cast<double>(getNumIterations()) : 0.0;
}

/// \brief Getter for the number of iterations since the last run() or initialize() call
std::size_t SamplerBase::Statistics::getNumIterations() const {
  return nIterations;
}

/// \brief Getter for the number of iterations since the last run() or initialize() call
std::size_t SamplerBase::Statistics::getNumAcceptedSamples(bool total) const {
  return total ? nSamplesAcceptedTotal : nSamplesAcceptedThisRun;
}

void SamplerBase::Statistics::updateWeightedMeanAcceptanceProbability(const double prob) {
  SM_ASSERT_GE_DBG(Exception, prob, 0.0, "");
  SM_ASSERT_LE_DBG(Exception, prob, 1.0, "");
  weightedMeanAcceptanceProbability += nIterations == 0 ? prob : -0.1 * (weightedMeanAcceptanceProbability - prob);
}

/// \brief Run the sampler for \p nSteps
void SamplerBase::run(const std::size_t nSteps) {

  if (!_problemManager.isInitialized())
    initialize();

  bool accepted;
  double accProb;
  _statistics.nSamplesAcceptedThisRun = 0;

  for (size_t cnt = 0; cnt < nSteps; cnt++) {

    step(accepted, accProb);

    // Update statistics
    if(accepted) {
      _statistics.nSamplesAcceptedThisRun++;
      _statistics.nSamplesAcceptedTotal++;
    }
    _statistics.nIterations++;
    _statistics.updateWeightedMeanAcceptanceProbability(accProb);
  }

  SM_VERBOSE_STREAM("Acceptance rate -- this run: " << fixed << setprecision(4) <<
                static_cast<double>(_statistics.nSamplesAcceptedThisRun)/nSteps << " (" << _statistics.nSamplesAcceptedThisRun << " of " << nSteps << "), total: " <<
                _statistics.getAcceptanceRate() << " (" << _statistics.getNumAcceptedSamples(true) << " of " << _statistics.getNumIterations() << "), mean acceptance probability: " <<
                _statistics.getWeightedMeanAcceptanceProbability());

}

/// \brief Set up to work on the log density. The log density may neglect the normalization constant.
void SamplerBase::setNegativeLogDensity(boost::shared_ptr<OptimizationProblemBase> negLogDensity) {
  _problemManager.setProblem(negLogDensity);
}

/// \brief Mutable getter for the log density formulation
boost::shared_ptr<OptimizationProblemBase> SamplerBase::getNegativeLogDensity() {
  return _problemManager.getProblem();
}

/// \brief Const getter for the log density formulation
boost::shared_ptr<const OptimizationProblemBase> SamplerBase::getNegativeLogDensity() const {
  return _problemManager.getProblem();
}

/// \brief Signal the sampler that the negative log density formulation changed.
void SamplerBase::signalNegativeLogDensityChanged() {
  _problemManager.signalProblemChanged();
}

/// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
///        hooked up to design variables and running finite differences on error terms where this is possible.
void SamplerBase::checkNegativeLogDensitySetup() const {
  _problemManager.checkProblemSetup();
}

/// \brief Evaluate the current negative log density
double SamplerBase::evaluateNegativeLogDensity() const {
  return _problemManager.evaluateError();
}

/// \brief Initialization method
void SamplerBase::initialize() {
  reset();
  _problemManager.initialize();
}

/// \brief Reset the sampler
void SamplerBase::reset() {
  _statistics.reset();
  resetImplementation();
}

/// \brief Const getter for statistics
const SamplerBase::Statistics& SamplerBase::statistics() const {
  return _statistics;
}

}
}
