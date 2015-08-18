/*
 * SamplerMetropolisHastings.cpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#include <aslam/backend/SamplerMetropolisHastings.hpp>

#include <cmath> // std::exp

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

std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerMetropolisHastingsOptions& options) {
  out << "SamplerMetropolisHastingsOptions:" << std::endl;
  out << "\ttransitionKernelSigma: " << options.transitionKernelSigma << std::endl;
  return out;
}



SamplerMetropolisHastings::SamplerMetropolisHastings() :
  _options() {

}

SamplerMetropolisHastings::SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options) :
  _options(options) {

}

void SamplerMetropolisHastings::runImplementation(const std::size_t nStepsMax, const std::size_t nAcceptedSamples, Statistics& statistics) {

  auto normal_dist = [&] (double) { return _options.transitionKernelSigma*sm::random::randn(); };

  double negLogDensity = evaluateNegativeLogDensity();

  for (; statistics.nIterationsThisRun < nStepsMax; statistics.nIterationsThisRun++) {

    if (statistics.nSamplesAcceptedThisRun >= nAcceptedSamples) {
      SM_FINE("Required number of accepted samples reached, terminating loop.");
      break;
    }

    const ColumnVectorType dx = ColumnVectorType::NullaryExpr(getProblemManager().numOptParameters(), normal_dist);
    getProblemManager().applyStateUpdate(dx);

    const double negLogDensityNew = evaluateNegativeLogDensity();

    const double acceptanceProbability = std::exp(std::min(0.0, -negLogDensityNew + negLogDensity));
    SM_VERBOSE_STREAM("NegLogDensity: " << negLogDensity << "->" << negLogDensityNew << ", acceptance probability: " << acceptanceProbability);

    if (sm::random::randLU(0.0, 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
      negLogDensity = negLogDensityNew;
      statistics.nSamplesAcceptedThisRun++;
      SM_VERBOSE_STREAM("Sample accepted");
    } else { // sample rejected, we revert the update
      getProblemManager().revertLastStateUpdate();
      SM_VERBOSE_STREAM("Sample rejected");
    }

  }

}

} /* namespace aslam */
} /* namespace backend */
