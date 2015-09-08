/*
 * SamplerHmc.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Ulrich Schwesinger
 */

#include <aslam/backend/SamplerHybridMcmc.hpp>
#include <aslam/backend/DesignVariable.hpp>

#include <cmath>

#include <sm/logging.hpp>
#include <sm/random.hpp>
#include <sm/PropertyTree.hpp>

using namespace std;

namespace aslam {
namespace backend {

SamplerHybridMcmcOptions::SamplerHybridMcmcOptions() {
  check();
}

SamplerHybridMcmcOptions::SamplerHybridMcmcOptions(const sm::PropertyTree& config) {

  initialLeapFrogStepSize = config.getDouble("initialLeapFrogStepSize", initialLeapFrogStepSize);
  minLeapFrogStepSize = config.getDouble("minLeapFrogStepSize", minLeapFrogStepSize);
  maxLeapFrogStepSize = config.getDouble("maxLeapFrogStepSize", maxLeapFrogStepSize);
  incFactorLeapFrogStepSize = config.getDouble("incFactorLeapFrogStepSize", incFactorLeapFrogStepSize);
  decFactorLeapFrogStepSize = config.getDouble("decFactorLeapFrogStepSize", decFactorLeapFrogStepSize);
  targetAcceptanceRate = config.getDouble("targetAcceptanceRate", targetAcceptanceRate);
  nLeapFrogSteps = config.getInt("nLeapFrogSteps", nLeapFrogSteps);
  nThreads = config.getDouble("nThreads", nThreads);

  check();

}

void SamplerHybridMcmcOptions::check() const {
  SM_ASSERT_GT(Exception, initialLeapFrogStepSize, 0.0, "");
  SM_ASSERT_GT(Exception, minLeapFrogStepSize, 0.0, "");
  SM_ASSERT_GT(Exception, maxLeapFrogStepSize, 0.0, "");
  SM_ASSERT_GE(Exception, maxLeapFrogStepSize, minLeapFrogStepSize, "");
  SM_ASSERT_GE(Exception, incFactorLeapFrogStepSize, 1.0, "");
  SM_ASSERT_GT(Exception, decFactorLeapFrogStepSize, 0.0, "");
  SM_ASSERT_LE(Exception, decFactorLeapFrogStepSize, 1.0, "");
  SM_ASSERT_GE(Exception, targetAcceptanceRate, 0.0, "");
  SM_ASSERT_LE(Exception, targetAcceptanceRate, 1.0, "");
}

ostream& operator<<(ostream& out, const aslam::backend::SamplerHybridMcmcOptions& options) {
  out << "SamplerHmcOptions:\n";
  out << "\tinitialLeapFrogStepSize: " << options.initialLeapFrogStepSize << endl;
  out << "\tminLeapFrogStepSize: " << options.minLeapFrogStepSize << endl;
  out << "\tmaxLeapFrogStepSize: " << options.maxLeapFrogStepSize << endl;
  out << "\tincFactorLeapFrogStepSize: " << options.incFactorLeapFrogStepSize << endl;
  out << "\tdecFactorLeapFrogStepSize: " << options.decFactorLeapFrogStepSize << endl;
  out << "\ttargetAcceptanceRate: " << options.targetAcceptanceRate << endl;
  out << "\tnLeapFrogSteps: " << options.nLeapFrogSteps << endl;
  out << "\tnThreads: " << options.nThreads << endl;
  return out;
}




SamplerHybridMcmc::SamplerHybridMcmc() :
  _options(),
  _gradient(),
  _u(),
  _lastSampleAccepted(false),
  _stepLength(_options.initialLeapFrogStepSize) {

}

SamplerHybridMcmc::SamplerHybridMcmc(const SamplerHybridMcmcOptions& options) :
  _options(options),
  _gradient(),
  _u(),
  _lastSampleAccepted(false),
  _stepLength(_options.initialLeapFrogStepSize) {

}

void SamplerHybridMcmc::saveDesignVariables() {
  Timer t("SamplerHmc: Save design variables", false);
  for (size_t i = 0; i < getProblemManager().numDesignVariables(); i++) {
    const DesignVariable* dv = getProblemManager().designVariable(i);
    dv->getParameters(_dvState[i].second);
  }
}

void SamplerHybridMcmc::revertUpdateDesignVariables() {
  Timer t("SamplerHmc: Revert update design variables", false);
  for (auto dvParamPair : _dvState)
    dvParamPair.first->setParameters(dvParamPair.second);
}

void SamplerHybridMcmc::initialize() {
  SamplerBase::initialize();
  _dvState.resize(getProblemManager().numDesignVariables());
  for (size_t i = 0; i < _dvState.size(); i++)
    _dvState[i].first = getProblemManager().designVariable(i);
  _gradient.resize(getProblemManager().numOptParameters());
}

void SamplerHybridMcmc::setOptions(const SamplerHybridMcmcOptions& options) {
  _options = options;
  _stepLength = _options.initialLeapFrogStepSize;
}

void SamplerHybridMcmc::step(bool& accepted, double& acceptanceProbability) {

  // Note: The notation follows the implementation here:
  // https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/

  using namespace Eigen;

  Timer timeGrad("SamplerHybridMcmc: Compute---Gradient", true);

  // Adapt step length
  if (statistics().getWeightedMeanAcceptanceProbability() > _options.targetAcceptanceRate)
    _stepLength *= _options.incFactorLeapFrogStepSize;
  else
    _stepLength *= _options.decFactorLeapFrogStepSize;
  _stepLength = max(min(_stepLength, _options.maxLeapFrogStepSize), _options.minLeapFrogStepSize); // clip

  // pre-computations for speed-up of upcoming calculations
  const double deltaHalf = _stepLength/2.;

  ColumnVectorType dxStar;
  ColumnVectorType pStar;
  double u0, k0, kStar, eTotal0, eTotalStar;

  // save the state of the design variables to be able to revert them later to this stage
  saveDesignVariables();

  // ******** Simulate Hamiltonian dynamics via Leap-Frog method ********* //

  // sample random momentum
  auto normal_dist = [&] (int) { return sm::random::randn(); };
  pStar = ColumnVectorType::NullaryExpr(getProblemManager().numOptParameters(), normal_dist);

  // evaluate energies at start of trajectory
  u0 = _lastSampleAccepted ? _u : evaluateNegativeLogDensity(); // potential energy
  SM_ASSERT_EQ_DBG(Exception, evaluateNegativeLogDensity(), u0, ""); // check that caching works
  k0 = 0.5*pStar.transpose()*pStar; // kinetic energy
  eTotal0 = u0 + k0;

  // first half step of momentum
  if (!_lastSampleAccepted) { // we can avoid recomputing the gradient if the last sample was accepted
    timeGrad.start();
    getProblemManager().computeGradient(_gradient, _options.nThreads, false /*TODO: useMEstimator*/);
    timeGrad.stop();
  }
#ifndef NDEBUG
  RowVectorType grad;
  getProblemManager().computeGradient(grad, _options.nThreads, false);
  SM_ASSERT_TRUE(Exception, _gradient.isApprox(grad), ""); // check that caching works
#endif
  RowVectorType gradient0 = _gradient; // to be able to restore later


  pStar -= deltaHalf*_gradient;

  // first full step for position/sample
  dxStar = _stepLength*pStar;
  SM_ASSERT_TRUE_DBG(Exception, pStar.allFinite(), "Position " << dxStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
      "Maybe the step length of the Leap-Frog method is too large.");
  getProblemManager().applyStateUpdate(dxStar);

  SM_ALL_STREAM("Step 0 -- Momentum: " << pStar.transpose() << ", position update: " << dxStar.transpose());

   // L-1 full steps
   for(size_t l = 1; l < _options.nLeapFrogSteps - 1; ++l) {

     // momentum
     timeGrad.start();
     getProblemManager().computeGradient(_gradient, _options.nThreads, false /*TODO: useMEstimator*/);
     timeGrad.stop();

     pStar -= _stepLength*_gradient;
     SM_ASSERT_TRUE_DBG(Exception, pStar.allFinite(), "Momentum " << pStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
         "Maybe the step length of the Leap-Frog method is too large.");

     // position/sample
     dxStar = _stepLength*pStar;
     SM_ASSERT_TRUE_DBG(Exception, dxStar.allFinite(), "Position " << dxStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
         "Maybe the step length of the Leap-Frog method is too large.");
     getProblemManager().applyStateUpdate(dxStar);

     SM_ALL_STREAM("Step " << l << " -- Momentum: " << pStar.transpose() << ", position update: " << dxStar.transpose());

   }

    // last half step
    timeGrad.start();
    getProblemManager().computeGradient(_gradient, _options.nThreads, false /*TODO: useMEstimator*/);
    timeGrad.stop();
    pStar -= deltaHalf*_gradient;

    // ******************************************************************* //

    // evaluate energies at end of trajectory
    _u = evaluateNegativeLogDensity(); // potential energy
    kStar = 0.5*pStar.transpose()*pStar; // kinetic energy
    eTotalStar = _u + kStar;


    // acceptance/rejection probability
    acceptanceProbability = 0.0; // this rejects the sample if the energy is not finite
    if (isfinite(eTotalStar)) {
      acceptanceProbability = min(1.0, exp(eTotal0 - eTotalStar)); // TODO: can we remove the thresholding?
      SM_FINEST_STREAM("Energy " << eTotal0 << " (potential: " << u0 << ", kinetic: " << k0 << ") ==> " <<
                       eTotalStar << " (potential: " << _u << ", kinetic: " << kStar << "), acceptanceProbability = " << acceptanceProbability);
    } else {
      SM_WARN_STREAM("Leap-Frog method diverged, reducing step length...");
      _stepLength *= _options.decFactorLeapFrogStepSize;
    }

    if (sm::random::randLU(0., 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
      SM_FINEST_STREAM("Sample accepted");
      accepted = _lastSampleAccepted = true;
    } else { // sample rejected, we revert the update
      revertUpdateDesignVariables();
      _u = u0;
      _gradient = gradient0;
      SM_FINEST_STREAM("Sample rejected");
      accepted = _lastSampleAccepted = false;
    }

}

} /* namespace aslam */
} /* namespace backend */
