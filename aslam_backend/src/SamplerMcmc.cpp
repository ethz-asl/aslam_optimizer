/*
 * SamplerMcmc.cpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#include <cmath>

#include <sm/logging.hpp>
#include <sm/random.hpp>

#include <aslam/backend/SamplerMcmc.hpp>

using namespace std;

namespace aslam {
namespace backend {

SamplerMcmcOptions::SamplerMcmcOptions() :
  transitionKernelSigma(0.1) {

}

SamplerMcmcOptions::SamplerMcmcOptions(const sm::PropertyTree& config) :
    transitionKernelSigma(config.getDouble("transitionKernelSigma", transitionKernelSigma)) {

}





SamplerMcmc::SamplerMcmc() :
  _options(),
  _numParameters(0),
  _isInitialized(false),
  _nIterations(0),
  _acceptanceRate(0.0) {

}

SamplerMcmc::SamplerMcmc(const SamplerMcmcOptions& options) :
  _options(options),
  _numParameters(0),
  _isInitialized(false),
  _nIterations(0),
  _acceptanceRate(0.0) {

}

void SamplerMcmc::setLogDensity(LogDensityPtr problem) {
  _problem = problem;
  _isInitialized = false;
}

void SamplerMcmc::initialize() {

  SM_ASSERT_FALSE(Exception, _problem == nullptr, "No log density has been set");
  SM_DEBUG_STREAM("SamplerMcmc: Initializing problem...");
  Timer init("SamplerMcmc: Initialize---Total", false);

  _designVariables.clear();
  _designVariables.reserve(_problem->numDesignVariables());
  Timer initDv("SamplerMcmc: Initialize---Design Variables", false);
  // Run through all design variables adding active ones to an active list.
  for (size_t i = 0; i < _problem->numDesignVariables(); ++i) {
    DesignVariable* const dv = _problem->designVariable(i);
    if (dv->isActive())
      _designVariables.push_back(dv);
  }
  SM_ASSERT_FALSE(Exception, _problem->numDesignVariables() > 0 && _designVariables.empty(),
                  "It is illegal to run the sampler with all marginalized design variables. Did you forget to set the design variables as active?");
  SM_ASSERT_FALSE(Exception, _designVariables.empty(), "It is illegal to run the optimizer with all marginalized design variables.");
  // Assign block indices to the design variables.
  for (size_t i = 0; i < _designVariables.size(); ++i) {
    _designVariables[i]->setBlockIndex(i);
    _designVariables[i]->setColumnBase(_numParameters);
    _numParameters += _designVariables[i]->minimalDimensions();
  }
  initDv.stop();

  _nIterations = 0;
  _acceptanceRate = 0.0;

  Timer initEt("SamplerMcmc: Initialize---Error Terms", false);
  // Get all of the error terms that work on these design variables.
  _errorTermsNS.clear();
  for (size_t i = 0; i < _problem->numNonSquaredErrorTerms(); ++i)
    _errorTermsNS.push_back(_problem->nonSquaredErrorTerm(i));
  _errorTermsS.clear();
  for (size_t i = 0; i < _problem->numErrorTerms(); ++i)
    _errorTermsS.push_back(_problem->errorTerm(i));
  initEt.stop();
  SM_ASSERT_FALSE(Exception, _errorTermsNS.empty() && _errorTermsS.empty(), "It is illegal to run the sampler with no error terms attached to the log density.");

  _isInitialized = true;

  SM_DEBUG_STREAM("SamplerMcmc: Initialized problem with " << _problem->numDesignVariables() << " design variable(s), " <<
                  _errorTermsNS.size() << " non-squared error term(s) and " << _errorTermsS.size() << " squared error term(s)");

}

void SamplerMcmc::updateDesignVariables() {
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

void SamplerMcmc::revertUpdateDesignVariables() {
  Timer t("SamplerMcmc: Revert update design variables", false);
  for (auto dv : _designVariables)
    dv->revertUpdate();
}

double SamplerMcmc::evaluateLogDensity() const {
  Timer t("SamplerMcmc: Compute---Log density", false);
  double logDensity = 0.0;
  for (auto e : _errorTermsS)
    logDensity += e->evaluateError();
  for (auto e : _errorTermsNS)
    logDensity += e->evaluateError();
  return logDensity;
}

void SamplerMcmc::run(const std::size_t nSteps) {

  if (!_isInitialized)
    initialize();

  double logDensity;
  if (nSteps > 0)
    logDensity = evaluateLogDensity();

  _acceptanceRate *= static_cast<double>(_nIterations);

  for (std::size_t cnt = 0; cnt < nSteps; cnt++, _nIterations++) {

    updateDesignVariables();
    const double logDensityNew = evaluateLogDensity();

    const double acceptanceProbability = std::exp(std::min(0.0, logDensityNew - logDensity));
    SM_VERBOSE_STREAM("LogDensity: " << logDensity << "->" << logDensityNew << ", acceptance probability: " << acceptanceProbability);

    if (sm::random::randLU(0., 1.0) < acceptanceProbability) {
      logDensity = logDensityNew;
      _acceptanceRate += 1.;
      SM_VERBOSE_STREAM("Sample accepted");
      // sample accepted, we keep the new design variables
    } else {
      revertUpdateDesignVariables();
      SM_VERBOSE_STREAM("Sample rejected");
      // sample rejected, we revert the update
    }

  }

  if (_nIterations > 0)
    _acceptanceRate /= static_cast<double>(_nIterations);

}

void SamplerMcmc::checkLogDensitySetup()
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
