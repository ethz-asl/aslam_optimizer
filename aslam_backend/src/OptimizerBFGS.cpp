#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/logging.hpp>

namespace aslam {
namespace backend {

OptimizerBFGSOptions::OptimizerBFGSOptions() {
  check();
}

OptimizerBFGSOptions::OptimizerBFGSOptions(const sm::PropertyTree& config) :
    convergenceGradientNorm(config.getDouble("convergenceGradientNorm", convergenceGradientNorm)),
    convergenceDx(config.getDouble("convergenceDx", convergenceDx)),
    maxIterations(config.getInt("maxIterations", maxIterations)),
    nThreads(config.getInt("nThreads", nThreads)),
    linesearch(config)
{
  check();
}

void OptimizerBFGSOptions::check() const {
  SM_ASSERT_GE( Exception, convergenceGradientNorm, 0.0, "");
  SM_ASSERT_GE( Exception, convergenceDx, 0.0, "");
  SM_ASSERT_TRUE( Exception, convergenceDx > 0 || convergenceGradientNorm > 0.0, "");
  SM_ASSERT_GE( Exception, maxIterations, -1, "");
  linesearch.check();
}

std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerBFGSOptions& options)
{
  out << "OptimizerBFGSOptions:\n";
  out << "\tconvergenceGradientNorm: " << options.convergenceGradientNorm << std::endl;
  out << "\tconvergenceDx: " << options.convergenceDx << std::endl;
  out << "\tmaxIterations: " << options.maxIterations << std::endl;
  out << "\tnThreads: " << options.nThreads << std::endl;
  out << options.linesearch << std::endl;
  return out;
}

void BFGSReturnValue::reset() {
  convergence = IN_PROGRESS;
  nIterations = nGradEvaluations = nObjectiveEvaluations = 0;
  gradientNorm = std::numeric_limits<double>::signaling_NaN();
  error = std::numeric_limits<double>::max();
}

bool BFGSReturnValue::success() const {
  return convergence != FAILURE && convergence != IN_PROGRESS;
}

bool BFGSReturnValue::failure() const {
  return convergence == FAILURE;
}

std::ostream& operator<<(std::ostream& out, const BFGSReturnValue::ConvergenceCriterion& convergence) {
  switch (convergence) {
    case BFGSReturnValue::ConvergenceCriterion::IN_PROGRESS:
      out << "IN_PROGRESS";
      break;
    case BFGSReturnValue::ConvergenceCriterion::FAILURE:
      out << "FAILURE";
      break;
    case BFGSReturnValue::ConvergenceCriterion::GRADIENT_NORM:
      out << "GRADIENT_NORM";
      break;
    case BFGSReturnValue::ConvergenceCriterion::DX:
      out << "DX";
      break;
  }
  return out;
}


OptimizerBFGS::OptimizerBFGS() :
    _options(OptimizerBFGSOptions()),
    _linesearch(this, _options.linesearch)
{
  _options.check();
  _linesearch.setEvaluateErrorCallback( boost::bind(&OptimizerBFGS::increaseEvaluateErrorCounter, this) );
  _linesearch.setEvaluateGradientCallback(boost::bind(&OptimizerBFGS::increaseEvaluateGradientCounter, this));
}

OptimizerBFGS::OptimizerBFGS(const OptimizerBFGSOptions& options) :
    _options(options),
    _linesearch(this, _options.linesearch)
{
  _options.check();
  _linesearch.setEvaluateErrorCallback( boost::bind(&OptimizerBFGS::increaseEvaluateErrorCounter, this) );
  _linesearch.setEvaluateGradientCallback(boost::bind(&OptimizerBFGS::increaseEvaluateGradientCounter, this));
}

OptimizerBFGS::OptimizerBFGS(const sm::PropertyTree& config) :
    _options(OptimizerBFGSOptions(config)),
    _linesearch(this, _options.linesearch)
{
  _options.check();
  _linesearch.setEvaluateErrorCallback( boost::bind(&OptimizerBFGS::increaseEvaluateErrorCounter, this) );
  _linesearch.setEvaluateGradientCallback(boost::bind(&OptimizerBFGS::increaseEvaluateGradientCounter, this));
}

OptimizerBFGS::~OptimizerBFGS()
{
}


void OptimizerBFGS::initialize()
{
  ProblemManager::initialize();
  reset();
}

void OptimizerBFGS::reset() {
  _returnValue.reset();
  _Hk = Eigen::MatrixXd::Identity(numOptParameters(), numOptParameters());
  _linesearch.initialize();
}

const BFGSReturnValue& OptimizerBFGS::optimize()
{
  Timer timeUpdateHessian("OptimizerBFGS: Update---Hessian", true);

  if (!isInitialized())
    initialize();

  using namespace Eigen;

  const MatrixXd I = MatrixXd::Identity(numOptParameters(), numOptParameters());
  RowVectorType gfk, gfkp1;
  gfk = _linesearch.getGradient();

  std::size_t cnt = 0;
  for (cnt = 0; _options.maxIterations == -1 || cnt < static_cast<size_t>(_options.maxIterations); ++cnt) {

    _returnValue.nIterations++;

    // compute search direction
    RowVectorType pk = -_Hk*gfk.transpose();
    _linesearch.setSearchDirection(pk);

    // perform line search
    if(!_linesearch.lineSearchWolfe12()) {
      _returnValue.convergence = BFGSReturnValue::FAILURE;
      break;
    }

    const double alpha_k = _linesearch.getCurrentStepLength();
    gfkp1 = _linesearch.getGradient();

    _returnValue.gradientNorm = gfk.norm();
    if (_returnValue.gradientNorm < _options.convergenceGradientNorm) {
      _returnValue.convergence = BFGSReturnValue::GRADIENT_NORM;
      SM_DEBUG_STREAM_NAMED("optimization", "BFGS: Current gradient norm " << _returnValue.gradientNorm <<
                            " is smaller than convergenceGradientNorm option -> terminating");
      break;
    }

    _returnValue.error = _linesearch.getError();
    if (!std::isfinite(_returnValue.error)) {
      _returnValue.convergence = BFGSReturnValue::FAILURE; // TODO: Is this really a failure?
      SM_WARN("BFGS: We correctly found +-inf as optimal value, or something went wrong?");
      break;
    }

    SM_DEBUG_STREAM_NAMED("optimization", "OptimizerBFGS: Iteration " << cnt << " -- Performed step with length " << alpha_k <<
                          ". New error: " << _returnValue.error << ", new gradient norm: " << _returnValue.gradientNorm);

    timeUpdateHessian.start();

    RowVectorType sk = alpha_k * pk;
    RowVectorType yk = gfkp1 - gfk;
    gfk = gfkp1;

    double rhok = 1./(yk*sk.transpose());
    if (std::isinf(rhok)) {
      rhok = 1000.0;
      SM_WARN("Divide-by-zero encountered: rhok assumed large");
    }

    MatrixXd C = sk.transpose() * yk * rhok;
    MatrixXd A1 = I - C;
    MatrixXd A2 = I - C.transpose();
    _Hk = A1 * (_Hk * A2) + (rhok * sk.transpose() * sk);

    timeUpdateHessian.stop();

  }

  if (!_returnValue.failure())
    SM_DEBUG_STREAM_NAMED("optimization", "OptimizerBFGS: Convergence " << _returnValue.convergence <<
                          " (iterations: " << _returnValue.nIterations << ", error: " << _returnValue.error <<
                          ", gradient norm: " << _returnValue.gradientNorm << ")");
  else
    SM_ERROR_STREAM("OptimizerBFGS: Failed to converge with status " << _returnValue.convergence << " (iterations: " <<
                    _returnValue.nIterations << ", error: " << _returnValue.error << ", gradient norm: " <<
                    _returnValue.gradientNorm << ")");

  return _returnValue;
}

void OptimizerBFGS::setOptions(const OptimizerBFGSOptions& options) {
  _options = options;
  _linesearch.options() = _options.linesearch;
}

} // namespace backend
} // namespace aslam
