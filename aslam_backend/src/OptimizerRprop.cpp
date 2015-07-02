#include <aslam/backend/OptimizerRprop.hpp>
// std::partial_sum
#include <numeric>
#include <aslam/backend/ErrorTerm.hpp>
// M.inverse()
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
//#include <sparse_block_matrix/linear_solver_dense.h>
//#include <sparse_block_matrix/linear_solver_cholmod.h>
#ifndef QRSOLVER_DISABLED
#include <sparse_block_matrix/linear_solver_spqr.h>
#include <aslam/backend/SparseQrLinearSystemSolver.hpp>
#endif
#include <aslam/backend/sparse_matrix_functions.hpp>
//#include <aslam/backend/BlockCholeskyLinearSystemSolver.hpp>
//#include <aslam/backend/SparseCholeskyLinearSystemSolver.hpp>
//#include <aslam/backend/DenseQrLinearSystemSolver.hpp>
#include <sm/PropertyTree.hpp>


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

OptimizerRprop::OptimizerRprop(const OptimizerRpropOptions& options) :
            _curr_gradient_norm(std::numeric_limits<double>::signaling_NaN()),
            _options(options),
            _numOptParameters(std::numeric_limits<std::size_t>::max()),
            _isInitialized(false)
{
  //  initializeLinearSolver();
  //  initializeTrustRegionPolicy();
}

OptimizerRprop::OptimizerRprop(const sm::PropertyTree& config, boost::shared_ptr<LinearSystemSolver> linearSystemSolver, boost::shared_ptr<TrustRegionPolicy> trustRegionPolicy) :
            _curr_gradient_norm(std::numeric_limits<double>::signaling_NaN()),
            _numOptParameters(std::numeric_limits<std::size_t>::max()),
            _isInitialized(false)
{
  OptimizerRpropOptions options;
  options.etaMinus = config.getDouble("etaMinus", options.etaMinus);
  options.etaPlus = config.getDouble("etaPlus", options.etaPlus);
  options.maxIterations = config.getInt("maxIterations", options.maxIterations);
  options.initialDelta = config.getDouble("initialDelta", options.initialDelta);
  options.verbose = config.getBool("verbose", options.verbose);
  options.maxDelta = config.getDouble("maxDelta", options.maxDelta);
  options.minDelta = config.getDouble("minDelta", options.minDelta);
  options.convergenceGradientNorm = config.getDouble("convergenceGradientNorm", options.convergenceGradientNorm);
  options.nThreads = config.getInt("nThreads", options.nThreads);
  options.linearSystemSolver = linearSystemSolver;
  options.trustRegionPolicy = trustRegionPolicy;
  _options = options;
}

OptimizerRprop::~OptimizerRprop()
{
}

/// \brief Set up to work on the optimization problem.
void OptimizerRprop::setProblem(boost::shared_ptr<OptimizationProblemBase> problem)
{
  _problem = problem;
  _isInitialized = false;
}

/// \brief initialize the optimizer to run on an optimization problem.
///        This should be called before calling optimize()
void OptimizerRprop::initialize()
{
  SM_ASSERT_FALSE(Exception, _problem == nullptr, "No optimization problem has been set");
  SM_ASSERT_EQ(Exception, _problem->numErrorTerms(), 0, "Cannot handle optimization problems with squared error terms");
  _options.verbose && std::cout << "Initializing problem..." << std::endl;
  Timer init("OptimizerRprop: Initialize Total");
  _designVariables.clear();
  _designVariables.reserve(_problem->numDesignVariables());
  _errorTerms.clear();
  _errorTerms.reserve(_problem->numNonSquaredErrorTerms());
  Timer initDv("OptimizerRprop: Initialize---Design Variables");
  // Run through all design variables adding active ones to an active list.
  for (size_t i = 0; i < _problem->numDesignVariables(); ++i) {
    DesignVariable* dv = _problem->designVariable(i);
    if (dv->isActive())
      _designVariables.push_back(dv);
  }
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

  _dx.resize(_numOptParameters, 1);
  _prev_gradient.resize(1, _numOptParameters);
  _delta = ColumnVectorType::Constant(_numOptParameters, _options.initialDelta);

  Timer initEt("OptimizerRprop: Initialize---Error Terms");
  // Get all of the error terms that work on these design variables.
  for (unsigned i = 0; i < _problem->numNonSquaredErrorTerms(); ++i) {
    NonSquaredErrorTerm* e = _problem->nonSquaredErrorTerm(i);
    _errorTerms.push_back(e);
  }
  initEt.stop();
  SM_ASSERT_FALSE(Exception, _errorTerms.empty(), "It is illegal to run the optimizer with no error terms.");
  _isInitialized = true;

  _options.verbose && std::cout <<  "Initialized problem with " << _problem->numDesignVariables() << " design variable(s) (" <<
      _numOptParameters << " optimization parameter(s)) and " << _errorTerms.size() << " error term(s)." << std::endl;

}

void OptimizerRprop::optimize()
{
  Timer timeErr("OptimizerRprop: evaluate error", true);
  Timer timeBackSub("OptimizerRprop: Back substitution", true);
  Timer timeSolve("OptimizerRprop: Solve linear system", true);
  // Select the design variables and (eventually) the error terms involved in the optimization.

  if (!_isInitialized)
    initialize();

  using namespace Eigen;

  bool isConverged = false;
  int cnt = 0;
  for (; _options.maxIterations == 0 || cnt < _options.maxIterations; ++cnt) {

    RowVectorType gradient;
    this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);

    SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient is not finite");

    _curr_gradient_norm = gradient.norm();

    if (_curr_gradient_norm < _options.convergenceGradientNorm) {
      isConverged = true;
      _options.verbose && std::cout << "Current gradient norm " << _curr_gradient_norm <<
          " is smaller than convergenceGradientNorm option -> terminating" << std::endl;
    }

    if (isConverged)
      break;

    _dx.setZero();
    for (std::size_t d = 0; d < _numOptParameters; ++d) {

      // Adapt delta
      if (_prev_gradient(d) * gradient(d) > 0.0)
        _delta(d) = std::min(_delta(d) * _options.etaPlus, _options.maxDelta);
      else if (_prev_gradient(d) * gradient(d) < 0.0)
        _delta(d) = std::max(_delta(d) * _options.etaMinus, _options.minDelta);

      // Compute design variable update vector
      if (gradient(d) > 0.0)
        _dx(d) -= _delta(d);
      else if (gradient(d) < 0.0)
        _dx(d) += _delta(d);

      // Set previous gradient
      if (_prev_gradient(d) * gradient(d) < 0.0)
        _prev_gradient(d) = 0.0;
      else
        _prev_gradient(d) = gradient(d);
    }

    if (_options.verbose) {
      std::cout << "Number of iterations: " << cnt << std::endl;
      std::cout << "\t gradient: " << gradient.format(IOFormat(StreamPrecision, 0, ", ", ", ", "", "", "[", "]")) << std::endl;
      std::cout << "\t dx:    " << _dx.format(IOFormat(StreamPrecision, 0, ", ", ", ", "", "", "[", "]"))  << std::endl;
      std::cout << "\t delta:    " << _delta.format(IOFormat(StreamPrecision, 0, ", ", ", ", "", "", "[", "]"))  << std::endl;
      std::cout << "\t norm:     " << _curr_gradient_norm << std::endl;
    }

    this->applyStateUpdate(_dx);

  }

  if (_options.verbose) {
    std::string convergence = isConverged ? "SUCCESS" : "FAILURE";
    std::cout << "Convergence: " << convergence << std::endl;
  }

}

DesignVariable* OptimizerRprop::designVariable(size_t i)
{
  SM_ASSERT_LT_DBG(Exception, i, _designVariables.size(), "index out of bounds");
  return _designVariables[i];
}

size_t OptimizerRprop::numDesignVariables() const
{
  return _designVariables.size();
}

double OptimizerRprop::applyStateUpdate(const ColumnVectorType& dx)
{
  // Apply the update to the dense state.
  int startIdx = 0;
  for (size_t i = 0; i < numDesignVariables(); i++) {
    DesignVariable* d = _designVariables[i];
    const int dbd = d->minimalDimensions();
    Eigen::VectorXd dxS = dx.segment(startIdx, dbd);
    dxS *= d->scaling();
    d->update(&dxS[0], dbd);
    startIdx += dbd;
  }
  // Track the maximum delta
  // \todo: should this be some other metric?
  double deltaX = dx.array().abs().maxCoeff();
  return deltaX;
}

void OptimizerRprop::revertLastStateUpdate()
{
  for (size_t i = 0; i < _designVariables.size(); i++)
    _designVariables[i]->revertUpdate();
}


OptimizerRpropOptions& OptimizerRprop::options()
{
  return _options;
}

void OptimizerRprop::printTiming() const
{
  sm::timing::Timing::print(std::cout);
}

void OptimizerRprop::checkProblemSetup()
{
  // Check that all error terms are hooked up to design variables.
  // TODO: Is this check really necessary? It's not wrong by default, but one could simply remove this error term.
  for (std::size_t i=0; i<_errorTerms.size(); i++)
    SM_ASSERT_GT(Exception, _errorTerms[i]->numDesignVariables(), 0, "Error term " << i << " has no design variables attached.");
}

void OptimizerRprop::computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator)
{
  SM_ASSERT_GT(Exception, nThreads, 0, "");
  std::vector<RowVectorType> gradients(nThreads, RowVectorType::Zero(1, _numOptParameters)); // compute gradients separately in different threads and add in the end
  boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job(boost::bind(&OptimizerRprop::evaluateGradients, this, _1, _2, _3, _4, _5));
  this->setupThreadedJob(job, nThreads, gradients, useMEstimator);
  // Add up the gradients
  outGrad = gradients[0];
  for (std::size_t i = 1; i<gradients.size(); i++)
    outGrad += gradients[i];
}

void OptimizerRprop::setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job, size_t nThreads, std::vector<RowVectorType>& out, bool useMEstimator)
{
  SM_ASSERT_GT(Exception, nThreads, 0, "");
  SM_ASSERT_EQ(Exception, nThreads, out.size(), "");
  if (nThreads == 1) {
    job(0, 0, _errorTerms.size(), useMEstimator, out[0]);
  } else {
    nThreads = std::min(nThreads, _errorTerms.size());
    // Give some error terms to each thread.
    std::vector<int> indices(nThreads + 1, 0);
    int nJPerThread = std::max(1, static_cast<int>(_errorTerms.size() / nThreads));
    for (unsigned i = 0; i < nThreads; ++i)
      indices[i + 1] = indices[i] + nJPerThread;
    // deal with the remainder.
    indices.back() = _errorTerms.size();
    // Build a thread pool and evaluate the jacobians.
    boost::thread_group threads;
    std::vector<SafeJob> jobs(nThreads);
    for (size_t i = 0; i < nThreads; ++i) {
      jobs[i] = SafeJob(boost::bind(job, i, indices[i], indices[i + 1], useMEstimator, out[i]));
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

void OptimizerRprop::evaluateGradients(size_t /* threadId */, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& J)
{
  for (size_t i = startIdx; i < endIdx; ++i) { // iterate through error terms
    JacobianContainer jc(1 /* dimension */);
    NonSquaredErrorTerm* e = _errorTerms[i];
    e->getWeightedJacobians(jc, useMEstimator);
    for (JacobianContainer::map_t::iterator it = jc.begin(); it != jc.end(); ++it) { // iterate over design variables of this error term
//      _options.verbose && std::cout << "J(0:" << it->second.rows()-1 << "," << it->first->columnBase() << ":" << it->first->columnBase() + it->second.cols() - 1 << ") += " << it->second << std::endl;
      J.block(0 /*e->rowBase()*/, it->first->columnBase(), it->second.rows(), it->second.cols()) += it->second;
//      _options.verbose && std::cout << "J(0:" << it->second.rows()-1 << "," << it->first->columnBase() << ":" << it->first->columnBase() + it->second.cols() - 1 << ") = " <<
//          J.block(0 /*e->rowBase()*/, it->first->columnBase(), it->second.rows(), it->second.cols()) << std::endl;
    }
  }
}


} // namespace backend
} // namespace aslam