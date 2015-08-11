#include <aslam/backend/util/ProblemManager.hpp>
#include <aslam/backend/OptimizationProblemBase.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>

#include <boost/thread.hpp>

#include <sm/logging.hpp>

namespace aslam {
namespace backend {

/// \brief The return value for a safe job
struct SafeJobReturnValue {
  SafeJobReturnValue(const std::exception& e) : _e(e) {}
  std::exception _e;
};

/// \brief Functor running a job catching all exceptions
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
      SM_FATAL_STREAM("Exception in thread block: " << e.what());
    }
  }
};

ProblemManager::ProblemManager() :
  _numOptParameters(0),
  _numErrorTerms(0),
  _isInitialized(false)
{

}

ProblemManager::~ProblemManager()
{
}

/// \brief Set up to work on the optimization problem.
void ProblemManager::setProblem(boost::shared_ptr<OptimizationProblemBase> problem)
{
  _problem = problem;
  _isInitialized = false;
}

/// \brief initialize the optimizer to run on an optimization problem.
///        This should be called before calling optimize()
void ProblemManager::initialize()
{

  SM_ASSERT_FALSE(Exception, _problem == nullptr, "No optimization problem has been set");
  Timer init("ProblemManager: Initialize total");
  _designVariables.clear();
  _designVariables.reserve(_problem->numDesignVariables());
  _errorTermsNS.clear();
  _errorTermsNS.reserve(_problem->numNonSquaredErrorTerms());
  _errorTermsS.clear();
  _errorTermsS.reserve(_problem->numErrorTerms());
  Timer initDv("ProblemManager: Initialize design Variables");
  // Run through all design variables adding active ones to an active list.
  for (size_t i = 0; i < _problem->numDesignVariables(); ++i) {
    DesignVariable* dv = _problem->designVariable(i);
    if (dv->isActive())
      _designVariables.push_back(dv);
  }
  SM_ASSERT_FALSE(Exception, _problem->numDesignVariables() > 0 && _designVariables.empty(),
                  "It is illegal to run the optimizer with all marginalized design variables. Did you forget to set the design variables as active?");
  SM_ASSERT_FALSE(Exception, _designVariables.empty(), "It is illegal to run the optimizer with all marginalized design variables.");
  // Assign block indices to the design variables.
  // "blocks" will hold the structure of the left-hand-side of Gauss-Newton
  _numOptParameters = 0;
  for (size_t i = 0; i < _designVariables.size(); ++i) {
    _designVariables[i]->setBlockIndex(i);
    _designVariables[i]->setColumnBase(_numOptParameters);
    _numOptParameters += _designVariables[i]->minimalDimensions();
  }
  initDv.stop();

  Timer initEt("ProblemManager: Initialize error terms");
  // Get all of the error terms that work on these design variables.
  _numErrorTerms = 0;
  for (unsigned i = 0; i < _problem->numNonSquaredErrorTerms(); ++i) {
    ScalarNonSquaredErrorTerm* e = _problem->nonSquaredErrorTerm(i);
    _errorTermsNS.push_back(e);
    _numErrorTerms++;
  }
  for (unsigned i = 0; i < _problem->numErrorTerms(); ++i) {
    ErrorTerm* e = _problem->errorTerm(i);
    _errorTermsS.push_back(e);
    _numErrorTerms++;
  }
  initEt.stop();
  SM_ASSERT_FALSE(Exception, _errorTermsNS.empty() && _errorTermsS.empty(), "It is illegal to run the optimizer with no error terms.");

  _isInitialized = true;

  SM_DEBUG_STREAM("ProblemManager: Initialized problem with " << _problem->numDesignVariables() <<
                  " design variable(s), " << _errorTermsNS.size() << " non-squared error term(s) and " <<
                  _errorTermsS.size() << " squared error term(s)");

}

DesignVariable* ProblemManager::designVariable(size_t i)
{
  SM_ASSERT_LT_DBG(Exception, i, _designVariables.size(), "index out of bounds");
  return _designVariables[i];
}

void ProblemManager::checkProblemSetup() const
{
  // Check that all error terms are hooked up to design variables.
  // TODO: Is this check really necessary? It's not wrong by default, but one could simply remove this error term.
  for (std::size_t i=0; i<_errorTermsNS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsNS[i]->numDesignVariables(), 0, "Non-squared error term " << i << " has no design variable(s) attached.");
  for (std::size_t i=0; i<_errorTermsS.size(); i++)
    SM_ASSERT_GT(Exception, _errorTermsS[i]->numDesignVariables(), 0, "Squared error term " << i << " has no design variable(s) attached.");
}

/**
 * Computes the gradient of the scalar objective function
 * @param[out] outGrad The gradient
 * @param nThreads How many threads to use
 * @param useMEstimator Whether to use an MEstimator
 */
void ProblemManager::computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator)
{
  SM_ASSERT_GT(Exception, nThreads, 0, "");
  Timer t("ProblemManager: Compute gradient", false);
  std::vector<RowVectorType> gradients(nThreads, RowVectorType::Zero(1, _numOptParameters)); // compute gradients separately in different threads and add in the end
  boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job(boost::bind(&ProblemManager::evaluateGradients, this, _1, _2, _3, _4, _5));
  this->setupThreadedJob(job, nThreads, gradients, useMEstimator);
  // Add up the gradients
  outGrad = gradients[0];
  for (std::size_t i = 1; i<gradients.size(); i++)
    outGrad += gradients[i];
}

double ProblemManager::evaluateError() const {
  Timer t("ProblemManager: Compute Negative Log density", false);
  double error = 0.0;
  for (auto e : _errorTermsS)
    error += e->evaluateError();
  for (auto e : _errorTermsNS)
    error += e->evaluateError();
  return error;
}

void ProblemManager::applyStateUpdate(const ColumnVectorType& dx)
{
  Timer t("ProblemManager: Apply state update", false);
  // Apply the update to the dense state.
  int startIdx = 0;
  for (size_t i = 0; i < _designVariables.size(); i++) {
    DesignVariable* d = _designVariables[i];
    const int dbd = d->minimalDimensions();
    Eigen::VectorXd dxS = dx.segment(startIdx, dbd);
    dxS *= d->scaling();
    d->update(&dxS[0], dbd);
    startIdx += dbd;
  }
}

void ProblemManager::revertLastStateUpdate()
{
  Timer t("ProblemManager: Revert last state update", false);
  for (size_t i = 0; i < _designVariables.size(); i++)
    _designVariables[i]->revertUpdate();
}

void ProblemManager::setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job, size_t nThreads, std::vector<RowVectorType>& out, bool useMEstimator)
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

/**
 * Evaluate the gradient of the objective function
 * @param
 * @param startIdx First error term index (including)
 * @param endIdx Last error term index (excluding)
 * @param useMEstimator Whether or not to use an MEstimator
 * @param J The gradient for the specified error terms
 */
void ProblemManager::evaluateGradients(size_t /* threadId */, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& J)
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

} // namespace backend
} // namespace aslam
