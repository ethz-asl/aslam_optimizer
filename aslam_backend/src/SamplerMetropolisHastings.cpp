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
  transitionKernelSigma(0.1),
  nThreadsEvaluateLogDensity(1) {

}

SamplerMetropolisHastingsOptions::SamplerMetropolisHastingsOptions(const sm::PropertyTree& config) :
    transitionKernelSigma(config.getDouble("transitionKernelSigma")),
    nThreadsEvaluateLogDensity(config.getDouble("nThreadsEvaluateLogDensity")) {

}

void SamplerMetropolisHastingsOptions::check() const {
  SM_ASSERT_GT( Exception, transitionKernelSigma, 0.0, "");
}

std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerMetropolisHastingsOptions& options) {
  out << "SamplerMetropolisHastingsOptions:" << std::endl;
  out << "\ttransitionKernelSigma: " << options.transitionKernelSigma << std::endl;
  out << "\tnThreadsEvaluateLogDensity: " << options.nThreadsEvaluateLogDensity << std::endl;
  return out;
}



SamplerMetropolisHastings::SamplerMetropolisHastings() :
  _options(),
  _negLogDensity(std::numeric_limits<double>::signaling_NaN()) {

}

SamplerMetropolisHastings::SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options) :
  _options(options),
  _negLogDensity(std::numeric_limits<double>::signaling_NaN()) {

}

void SamplerMetropolisHastings::step(bool& accepted, double& acceptanceProbability) {

  auto normal_dist = [&] (double) { return _options.transitionKernelSigma*sm::random::randn(); };

  if (isRecomputationNegLogDensityNecessary())
    _negLogDensity = evaluateNegativeLogDensity(_options.nThreadsEvaluateLogDensity);
#ifndef NDEBUG
  else
    SM_ASSERT_EQ(Exception, evaluateNegativeLogDensity(_options.nThreadsEvaluateLogDensity), _negLogDensity, ""); // check that caching works
#endif

  const ColumnVectorType dx = ColumnVectorType::NullaryExpr(getProblemManager().numOptParameters(), normal_dist);
  getProblemManager().applyStateUpdate(dx);

  const double negLogDensityNew = evaluateNegativeLogDensity(_options.nThreadsEvaluateLogDensity);

  acceptanceProbability = std::exp(std::min(0.0, -negLogDensityNew + _negLogDensity));
  SM_VERBOSE_STREAM("NegLogDensity: " << _negLogDensity << "->" << negLogDensityNew << ", acceptance probability: " << acceptanceProbability);

  if (sm::random::randLU(0.0, 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
    _negLogDensity = negLogDensityNew;
    accepted = true;
    SM_VERBOSE_STREAM("Sample accepted");
  } else { // sample rejected, we revert the update
    getProblemManager().revertLastStateUpdate();
    accepted = false;
    SM_VERBOSE_STREAM("Sample rejected");
  }

}

void SamplerMetropolisHastings::resetImplementation() {
  _negLogDensity = std::numeric_limits<double>::signaling_NaN();
}

} /* namespace aslam */
} /* namespace backend */
