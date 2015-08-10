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
  _isInitialized(false),
  _nIterations(0),
  _nSamplesAccepted(0) {

}

SamplerMetropolisHastings::SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options) :
  _options(options),
  _isInitialized(false),
  _nIterations(0),
  _nSamplesAccepted(0) {

}

void SamplerMetropolisHastings::setNegativeLogDensity(NegativeLogDensityPtr negLogDensity) {
  _negLogDensity = negLogDensity;
  _isInitialized = false;
}

void SamplerMetropolisHastings::initialize() {

  SM_ASSERT_FALSE(Exception, _negLogDensity == nullptr, "No negative log density has been set");
  SM_DEBUG_STREAM("SamplerMcmc: Initializing problem...");
  Timer init("SamplerMcmc: Initialize---Total", false);

  _designVariables.clear();
  _designVariables.reserve(_negLogDensity->numDesignVariables());
  Timer initDv("SamplerMcmc: Initialize---Design Variables", false);
  // Run through all design variables adding active ones to an active list.
  for (size_t i = 0; i < _negLogDensity->numDesignVariables(); ++i) {
    DesignVariable* const dv = _negLogDensity->designVariable(i);
    if (dv->isActive())
      _designVariables.push_back(dv);
  }
  SM_ASSERT_FALSE(Exception, _negLogDensity->numDesignVariables() > 0 && _designVariables.empty(),
                  "It is illegal to run the sampler with all marginalized design variables. Did you forget to set the design variables as active?");
  SM_ASSERT_FALSE(Exception, _designVariables.empty(), "It is illegal to run the optimizer with all marginalized design variables.");
  // Assign block indices to the design variables.
  size_t colBase = 0;
  for (size_t i = 0; i < _designVariables.size(); ++i) {
    _designVariables[i]->setBlockIndex(i);
    _designVariables[i]->setColumnBase(colBase);
    colBase += _designVariables[i]->minimalDimensions();
  }
  initDv.stop();

  _nIterations = 0;
  _nSamplesAccepted = 0;

  Timer initEt("SamplerMcmc: Initialize---Error Terms", false);
  // Get all of the error terms that work on these design variables.
  _errorTermsNS.clear();
  _errorTermsNS.reserve(_negLogDensity->numNonSquaredErrorTerms());
  for (size_t i = 0; i < _negLogDensity->numNonSquaredErrorTerms(); ++i)
    _errorTermsNS.push_back(_negLogDensity->nonSquaredErrorTerm(i));
  _errorTermsS.clear();
  _errorTermsNS.reserve(_negLogDensity->numErrorTerms());
  for (size_t i = 0; i < _negLogDensity->numErrorTerms(); ++i)
    _errorTermsS.push_back(_negLogDensity->errorTerm(i));
  initEt.stop();
  SM_ASSERT_FALSE(Exception, _errorTermsNS.empty() && _errorTermsS.empty(), "It is illegal to run the sampler with no error terms attached to the negative log density.");

  _isInitialized = true;

  SM_DEBUG_STREAM("SamplerMcmc: Initialized problem with " << _negLogDensity->numDesignVariables() << " design variable(s), " <<
                  _errorTermsNS.size() << " non-squared error term(s) and " << _errorTermsS.size() << " squared error term(s)");

}

void SamplerMetropolisHastings::updateDesignVariables() {
  Timer t("SamplerMcmc: Update design variables", false);
  for (auto dv : _designVariables) {
    const int dim = dv->minimalDimensions();
    Eigen::VectorXd dvDx(dim);
    for (int i=0; i<dvDx.rows(); ++i)
      dvDx[i] = _options.transitionKernelSigma*sm::random::randn(); // evaluate Gaussian transition kernel
    dvDx *= dv->scaling();
    dv->update(&dvDx(0), dim);
  }
}

void SamplerMetropolisHastings::revertUpdateDesignVariables() {
  Timer t("SamplerMcmc: Revert update design variables", false);
  for (auto dv : _designVariables)
    dv->revertUpdate();
}

double SamplerMetropolisHastings::evaluateNegativeLogDensity() const {
  Timer t("SamplerMcmc: Compute---Negative Log density", false);
  double negLogDensity = 0.0;
  for (auto e : _errorTermsS)
    negLogDensity += e->evaluateError();
  for (auto e : _errorTermsNS)
    negLogDensity += e->evaluateError();
  return negLogDensity;
}

void SamplerMetropolisHastings::run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples /* = std::numeric_limits<std::size_t>::max() */) {

  SM_ASSERT_GT(Exception, nStepsMax, 0, "It does not make sense to run the sampler with no steps.");
  SM_ASSERT_GT(Exception, nAcceptedSamples, 0, "It does not make sense to run the sampler until zero samples were accepted.");

  _nSamplesAccepted = 0;

  if (!_isInitialized)
    initialize();

  double negLogDensity = evaluateNegativeLogDensity();

  for (_nIterations = 0; _nIterations < nStepsMax; _nIterations++) {

    if (_nSamplesAccepted >= nAcceptedSamples) {
      SM_DEBUG("Required number of accepted samples reached, terminating loop.");
      break;
    }

    updateDesignVariables();
    const double negLogDensityNew = evaluateNegativeLogDensity();

    const double acceptanceProbability = std::exp(std::min(0.0, -negLogDensityNew + negLogDensity));
    SM_VERBOSE_STREAM("NegLogDensity: " << negLogDensity << "->" << negLogDensityNew << ", acceptance probability: " << acceptanceProbability);

    if (sm::random::randLU(0., 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
      negLogDensity = negLogDensityNew;
      _nSamplesAccepted++;
      SM_VERBOSE_STREAM("Sample accepted");
    } else { // sample rejected, we revert the update
      revertUpdateDesignVariables();
      SM_VERBOSE_STREAM("Sample rejected");
    }

  }

}

void SamplerMetropolisHastings::checkNegativeLogDensitySetup()
{
  // Check that all error terms are hooked up to design variables.
  // TODO: Is this check really necessary? It's not wrong by default, but one could simply remove this error term.
  for (std::size_t i=0; i<_errorTermsNS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsNS[i]->numDesignVariables(), 0, "Non-squared error term " << i << " has no design variable(s) attached.");
  for (std::size_t i=0; i<_errorTermsS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsS[i]->numDesignVariables(), 0, "Squared error term " << i << " has no design variable(s) attached.");
}

} /* namespace aslam */
} /* namespace backend */
