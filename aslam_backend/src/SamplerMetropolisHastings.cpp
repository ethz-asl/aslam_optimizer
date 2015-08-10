/*
 * SamplerMcmc.cpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#include <aslam/backend/SamplerMetropolisHastings.hpp>
#include <cmath>

#include <sm/logging.hpp>
#include <sm/random.hpp>


using namespace std;

namespace aslam {
namespace backend {

SamplerMetropolisHastingsOptions::SamplerMetropolisHastingsOptions() :
  transitionKernelSigma(0.1) {

}

SamplerMetropolisHastingsOptions::SamplerMetropolisHastingsOptions(const sm::PropertyTree& config) :
    transitionKernelSigma(config.getDouble("transitionKernelSigma", transitionKernelSigma)) {

}





SamplerMetropolisHastings::SamplerMetropolisHastings() :
  _options(),
  _nIterations(0),
  _nSamplesAccepted(0) {

}

SamplerMetropolisHastings::SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options) :
  _options(options),
  _nIterations(0),
  _nSamplesAccepted(0) {

}

void SamplerMetropolisHastings::initializeImplementation() {
  _nIterations = 0;
  _nSamplesAccepted = 0;
}

void SamplerMetropolisHastings::runImplementation(const std::size_t nStepsMax, const std::size_t nAcceptedSamples) {

  auto normal_dist = [&] (double) { return _options.transitionKernelSigma*sm::random::randn(); };

  _nSamplesAccepted = 0;

  double negLogDensity = evaluateNegativeLogDensity();

  for (_nIterations = 0; _nIterations < nStepsMax; _nIterations++) {

    if (_nSamplesAccepted >= nAcceptedSamples) {
      SM_DEBUG("Required number of accepted samples reached, terminating loop.");
      break;
    }

    const ColumnVectorType dx = ColumnVectorType::NullaryExpr(numOptParameters(), normal_dist);
    applyStateUpdate(dx);

    const double negLogDensityNew = evaluateNegativeLogDensity();

    const double acceptanceProbability = std::exp(std::min(0.0, -negLogDensityNew + negLogDensity));
    SM_VERBOSE_STREAM("NegLogDensity: " << negLogDensity << "->" << negLogDensityNew << ", acceptance probability: " << acceptanceProbability);

    if (sm::random::randLU(0., 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
      negLogDensity = negLogDensityNew;
      _nSamplesAccepted++;
      SM_VERBOSE_STREAM("Sample accepted");
    } else { // sample rejected, we revert the update
      revertLastStateUpdate();
      SM_VERBOSE_STREAM("Sample rejected");
    }

  }

}

} /* namespace aslam */
} /* namespace backend */
