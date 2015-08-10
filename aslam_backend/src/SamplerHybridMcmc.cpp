/*
 * SamplerHmc.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Ulrich Schwesinger
 */

#include <aslam/backend/SamplerHybridMcmc.hpp>
#include <cmath>

#include <sm/logging.hpp>
#include <sm/random.hpp>


using namespace std;

namespace aslam {
namespace backend {

struct SafeJobReturnValue {
  SafeJobReturnValue(const std::exception& e) : _e(e) {}
  std::exception _e;
};
struct SafeJob {
  boost::function<void()> _fn;
  SafeJobReturnValue* _rval;
  SafeJob() : _rval(NULL) {}
  SafeJob(boost::function<void()> fn) : _fn(fn), _rval(NULL) {}
  ~SafeJob() {
    if (_rval) delete _rval;
  }

  void operator()() {
    try {
      _fn();
    } catch (const std::exception& e) {
      _rval = new SafeJobReturnValue(e);
      std::cout << "Exception in thread block: " << e.what() << std::endl;
    }
  }
};



SamplerHmcOptions::SamplerHmcOptions() :
  delta(0.1),
  nHamiltonianSteps(20),
  nThreads(4) {

  check();

}

SamplerHmcOptions::SamplerHmcOptions(const sm::PropertyTree& config) :
  delta(config.getDouble("delta", delta)),
  nHamiltonianSteps(config.getInt("nHamiltonianSteps", nHamiltonianSteps)),
  nThreads(config.getDouble("nThreads", nThreads)) {

  check();

}

void SamplerHmcOptions::check() const {
  SM_WARN_STREAM_COND(delta > 5.0, "Delta value for Leap-Frog algorithm very high!");
}

SamplerHmc::Statistics::Statistics() :
  nIterationsThisRun(0),
  nIterationsTotal(0),
  nSamplesAcceptedThisRun(0),
  nSamplesAcceptedTotal(0) {

}

void SamplerHmc::Statistics::reset() {
  nIterationsThisRun = 0;
  nIterationsTotal = 0;
  nSamplesAcceptedThisRun = 0;
  nSamplesAcceptedTotal = 0;
}




SamplerHmc::SamplerHmc() :
  _options(),
  _numOptParameters(0),
  _numErrorTerms(0),
  _isInitialized(false) {

}

SamplerHmc::SamplerHmc(const SamplerHmcOptions& options) :
  _options(options),
  _numOptParameters(0),
  _numErrorTerms(0),
  _isInitialized(false) {

}

void SamplerHmc::setNegativeLogDensity(NegativeLogDensityPtr negLogDensity) {
  _negLogDensity = negLogDensity;
  _isInitialized = false;
}

void SamplerHmc::initialize() {

  SM_ASSERT_FALSE(Exception, _negLogDensity == nullptr, "No negative log density has been set");
  SM_DEBUG_STREAM("SamplerHmc: Initializing problem...");
  Timer init("SamplerHmc: Initialize---Total", false);

  _designVariables.clear();
  _designVariables.reserve(_negLogDensity->numDesignVariables());
  Timer initDv("SamplerHmc: Initialize---Design Variables", false);
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
  _numOptParameters = 0;
  for (size_t i = 0; i < _designVariables.size(); ++i) {
    _designVariables[i]->setBlockIndex(i);
    _designVariables[i]->setColumnBase(_numOptParameters);
    _numOptParameters += _designVariables[i]->minimalDimensions();
  }
  initDv.stop();

  _statistics.reset();

  Timer initEt("SamplerHmc: Initialize---Error Terms", false);
  // Get all of the error terms that work on these design variables.
  _errorTermsNS.clear();
  _errorTermsNS.reserve(_negLogDensity->numNonSquaredErrorTerms());
  for (size_t i = 0; i < _negLogDensity->numNonSquaredErrorTerms(); ++i)
    _errorTermsNS.push_back(_negLogDensity->nonSquaredErrorTerm(i));
  _errorTermsS.clear();
  _errorTermsNS.reserve(_negLogDensity->numErrorTerms());
  for (size_t i = 0; i < _negLogDensity->numErrorTerms(); ++i)
    _errorTermsS.push_back(_negLogDensity->errorTerm(i));
  _numErrorTerms = _errorTermsS.size() + _errorTermsNS.size();
  initEt.stop();
  SM_ASSERT_FALSE(Exception, _errorTermsNS.empty() && _errorTermsS.empty(), "It is illegal to run the sampler with no error terms attached to the negative log density.");

  _isInitialized = true;

  SM_DEBUG_STREAM("SamplerHmc: Initialized problem with " << _negLogDensity->numDesignVariables() << " design variable(s), " <<
                  _errorTermsNS.size() << " non-squared error term(s) and " << _errorTermsS.size() << " squared error term(s)");

}

void SamplerHmc::updateDesignVariables(const ColumnVectorType& dx) {
  int startIdx = 0;
  for (size_t i = 0; i < _designVariables.size(); i++) {
    DesignVariable* d = _designVariables[i];
    const int dim = d->minimalDimensions();
    Eigen::VectorXd dxS = dx.segment(startIdx, dim);
    dxS *= d->scaling();
    d->update(&dxS[0], dim);
    startIdx += dim;
  }
}

void SamplerHmc::saveDesignVariables() {

  Timer t("SamplerHmc: Save design variables", false);

  _dvState.clear();
  _dvState.reserve(_designVariables.size());

  for (size_t i = 0; i < _designVariables.size(); i++) {
    DesignVariable* dv = _designVariables[i];
    Eigen::MatrixXd p;
    dv->getParameters(p);
    _dvState.push_back( std::make_pair(dv, p) );
  }
}

void SamplerHmc::revertUpdateDesignVariables() {
  Timer t("SamplerHmc: Revert update design variables", false);
  for (auto dvParamPair : _dvState)
    dvParamPair.first->setParameters(dvParamPair.second);
}

double SamplerHmc::evaluateNegativeLogDensity() const {
  Timer t("SamplerHmc: Compute---Negative Log density", false);
  double negLogDensity = 0.0;
  for (auto e : _errorTermsS)
    negLogDensity += e->evaluateError();
  for (auto e : _errorTermsNS)
    negLogDensity += e->evaluateError();
  SM_ASSERT_TRUE_DBG(Exception, std::isfinite(negLogDensity), "Negative log density " << negLogDensity << " is not finite!");
  return negLogDensity;
}

void SamplerHmc::run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples /* = std::numeric_limits<std::size_t>::max() */) {

  // Note: The notation follows the implementation here:
  // https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/

  using namespace Eigen;

  SM_ASSERT_GT(Exception, nStepsMax, 0, "It does not make sense to run the sampler with no steps.");
  SM_ASSERT_GT(Exception, nAcceptedSamples, 0, "It does not make sense to run the sampler until zero samples were accepted.");

  Timer timeGrad("SamplerHmc: Compute---Gradient", true);

  if (!_isInitialized)
    initialize();

  // pre-computations for speed-up of upcoming calculations
  const double deltaHalf = _options.delta/2.;
  auto normal_dist = [&] (double) { return sm::random::randn(); };

  ColumnVectorType dxStar;
  ColumnVectorType pStar;
  RowVectorType gradient;
  double U0, UStar, K0, KStar, ETotal0, ETotalStar;
  U0 = UStar = K0 = KStar = ETotal0 = ETotalStar = 0.0;
  bool lastSampleAccepted = false;

  _statistics.nSamplesAcceptedThisRun = 0;
  for (_statistics.nIterationsThisRun=0; _statistics.nIterationsThisRun < nStepsMax; _statistics.nIterationsThisRun++) {

    if (_statistics.nSamplesAcceptedThisRun >= nAcceptedSamples) {
      SM_DEBUG("Required number of accepted samples reached, terminating loop.");
      break;
    }

    // save the state of the design variables to be able to revert them later to this stage
    saveDesignVariables();

    // ******** Simulate Hamiltonian dynamics via Leap-Frog method ********* //

    // sample random momentum
    pStar = ColumnVectorType::NullaryExpr(_numOptParameters, normal_dist);

    // evaluate energies at start of trajectory
    U0 = lastSampleAccepted ? UStar : evaluateNegativeLogDensity(); // potential energy
    K0 = 0.5*pStar.transpose()*pStar; // kinetic energy
    ETotal0 = U0 + K0;

    // first half step of momentum
    if (!lastSampleAccepted) { // we can avoid recomputing the gradient if the last sample was accepted
      timeGrad.start();
      this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);
      timeGrad.stop();
    }

    pStar -= deltaHalf*gradient;

    // first full step for position/sample
    dxStar = _options.delta*pStar;
    SM_ASSERT_TRUE_DBG(Exception, pStar.allFinite(), "Position " << dxStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
        "Maybe the step length delta of the Leap-Frog simulation is too large.");
    updateDesignVariables(dxStar);

    SM_ALL_STREAM("Step 0 -- Momentum: " << pStar.transpose() << ", position update: " << dxStar.transpose());

     // L-1 full steps
     for(size_t l = 1; l < _options.nHamiltonianSteps - 1; ++l) {

       // momentum
       timeGrad.start();
       this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);
       timeGrad.stop();

       pStar -= _options.delta*gradient;
       SM_ASSERT_TRUE_DBG(Exception, pStar.allFinite(), "Momentum " << pStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
           "Maybe the step length delta of the Leap-Frog simulation is too large.");

       // position/sample
       dxStar = _options.delta*pStar;
       SM_ASSERT_TRUE_DBG(Exception, pStar.allFinite(), "Position " << dxStar.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite. "
           "Maybe the step length delta of the Leap-Frog simulation is too large.");
       updateDesignVariables(dxStar);

       SM_ALL_STREAM("Step " << l << " -- Momentum: " << pStar.transpose() << ", position update: " << dxStar.transpose());

     }

      // last half step
      timeGrad.start();
      this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);
      timeGrad.stop();
      pStar -= deltaHalf*gradient;

      // ******************************************************************* //

      // evaluate energies at end of trajectory
      UStar = evaluateNegativeLogDensity(); // potential energy
      KStar = 0.5*pStar.transpose()*pStar; // kinetic energy
      ETotalStar = UStar + KStar;
      SM_ASSERT_TRUE(Exception, std::isfinite(ETotalStar), "Total energy at the end is not finite. "
          "Maybe the step length delta of the Leap-Frog simulation is too large.");

      // acceptance/rejection probability
      const double acceptanceProbability = std::min(1.0, exp(ETotal0 - ETotalStar)); // TODO: can we remove the thresholding?
      SM_FINEST_STREAM("Energy " << ETotal0 << " (potential: " << U0 << ", kinetic: " << K0 << ") ==> " <<
                       ETotalStar << " (potential: " << UStar << ", kinetic: " << KStar << "), acceptanceProbability = " << acceptanceProbability);

      if (sm::random::randLU(0., 1.0) < acceptanceProbability) { // sample accepted, we keep the new design variables
        _statistics.nSamplesAcceptedThisRun++;
        SM_FINEST_STREAM("Sample accepted");
        lastSampleAccepted = true;
      } else { // sample rejected, we revert the update
        revertUpdateDesignVariables();
        SM_FINEST_STREAM("Sample rejected");
        lastSampleAccepted = false;
      }

  }

  _statistics.nSamplesAcceptedTotal += _statistics.nSamplesAcceptedThisRun;
  _statistics.nIterationsTotal += _statistics.nIterationsThisRun;

  SM_VERBOSE_STREAM("Acceptance rate -- this run: " << fixed << setprecision(4) <<
                _statistics.getAcceptanceRate(false) << " (" << _statistics.getNumAcceptedSamples(false) << " of " << _statistics.getNumIterations(false) << "), total: " <<
                _statistics.getAcceptanceRate(true) << " (" << _statistics.getNumAcceptedSamples(true) << " of " << _statistics.getNumIterations(true) << ")");

}

void SamplerHmc::checkNegativeLogDensitySetup()
{
  // Check that all error terms are hooked up to design variables.
  // TODO: Is this check really necessary? It's not wrong by default, but one could simply remove this error term.
  for (std::size_t i=0; i<_errorTermsNS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsNS[i]->numDesignVariables(), 0, "Non-squared error term " << i << " has no design variable(s) attached.");
  for (std::size_t i=0; i<_errorTermsS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsS[i]->numDesignVariables(), 0, "Squared error term " << i << " has no design variable(s) attached.");
}

void SamplerHmc::computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator)
{
  SM_ASSERT_GT(Exception, nThreads, 0, "");
  std::vector<RowVectorType> gradients(nThreads, RowVectorType::Zero(1, _numOptParameters)); // compute gradients separately in different threads and add in the end
  boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job(boost::bind(&SamplerHmc::evaluateGradients, this, _1, _2, _3, _4, _5));
  this->setupThreadedJob(job, nThreads, gradients, useMEstimator);
  // Add up the gradients
  outGrad = gradients[0];
  for (std::size_t i = 1; i<gradients.size(); i++)
    outGrad += gradients[i];
  SM_ASSERT_TRUE_DBG(Exception, outGrad.allFinite (), "Gradient " << outGrad.format(Eigen::IOFormat(2, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite");
}

void SamplerHmc::setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job, size_t nThreads, std::vector<RowVectorType>& out, bool useMEstimator)
{
  SM_ASSERT_GT(Exception, nThreads, 0, "");
  SM_ASSERT_EQ(Exception, nThreads, out.size(), "");
  if (nThreads == 1) {
    job(0, 0, _numErrorTerms, useMEstimator, out[0]);
  } else {
    nThreads = std::min(nThreads, _numErrorTerms);
    // Give some error terms to each thread.
    std::vector<int> indices(nThreads + 1, 0);
    int nJPerThread = std::max(1, static_cast<int>(_numErrorTerms / nThreads));
    for (unsigned i = 0; i < nThreads; ++i)
      indices[i + 1] = indices[i] + nJPerThread;
    // deal with the remainder.
    indices.back() = _numErrorTerms;
    // Build a thread pool and evaluate the jacobians.
    boost::thread_group threads;
    std::vector<SafeJob> jobs(nThreads);
    for (size_t i = 0; i < nThreads; ++i) {
      SM_DEBUG_STREAM("SamplerHmc: Creating thread no " << i << " to compute error terms [" <<
          indices[i] << "," << indices[i + 1] << ")");
      jobs[i] = SafeJob(boost::bind(job, i, indices[i], indices[i + 1], useMEstimator, boost::ref(out[i])));
      threads.create_thread(boost::ref(jobs[i]));
    }
    threads.join_all();
    // Now go through and look for exceptions.
    for (size_t i = 0; i < nThreads; ++i) {
      if (jobs[i]._rval != nullptr)
        throw jobs[i]._rval->_e;
    }
  }
}

void SamplerHmc::evaluateGradients(size_t /* threadId */, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& J)
{
  SM_ASSERT_LE_DBG(Exception, endIdx, _numErrorTerms, "");
  for (size_t i = startIdx; i < endIdx; ++i) { // iterate through error terms
    if (i < _errorTermsNS.size()) {
      JacobianContainer jc(1 /* dimension */);
      ScalarNonSquaredErrorTerm* e = _errorTermsNS[i];
      e->evaluateJacobians(jc, useMEstimator);
      for (JacobianContainer::map_t::iterator it = jc.begin(); it != jc.end(); ++it) // iterate over design variables of this error term
        J.block(0 /*e->rowBase()*/, it->first->columnBase(), it->second.rows(), it->second.cols()) += it->second;
    } else {
      ErrorTerm* e = _errorTermsS[i - _errorTermsNS.size()];
      JacobianContainer jc(e->dimension());
      e->getWeightedJacobians(jc, useMEstimator);
      ColumnVectorType ev;
      e->updateRawSquaredError();
      e->getWeightedError(ev, useMEstimator);
      for (JacobianContainer::map_t::iterator it = jc.begin(); it != jc.end(); ++it) {// iterate over design variables of this error term
        RowVectorType grad = 2.0*ev.transpose()*it->second;
        J.block(0 /*e->rowBase()*/, it->first->columnBase(), grad.rows(), grad.cols()) += grad;
      }
    }
  }
}

} /* namespace aslam */
} /* namespace backend */
