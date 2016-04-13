#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/logging.hpp>

namespace aslam {
namespace backend {

OptimizerRpropOptions::OptimizerRpropOptions() {
  check();
}

OptimizerRpropOptions::OptimizerRpropOptions(const sm::PropertyTree& config) :
    etaMinus(config.getDouble("etaMinus", etaMinus)),
    etaPlus(config.getDouble("etaPlus", etaPlus)),
    initialDelta(config.getDouble("initialDelta", initialDelta)),
    minDelta(config.getDouble("minDelta", minDelta)),
    maxDelta(config.getDouble("maxDelta", maxDelta)),
    convergenceGradientNorm(config.getDouble("convergenceGradientNorm", convergenceGradientNorm)),
    convergenceDx(config.getDouble("convergenceDx", convergenceDx)),
    convergenceDObjective(config.getDouble("convergenceDObjective", convergenceDObjective)),
    maxIterations(config.getInt("maxIterations", maxIterations)),
    nThreads(config.getInt("nThreads", nThreads))
{
  check();
}

void OptimizerRpropOptions::check() const {
  SM_ASSERT_GT( Exception, etaMinus, 0.0, "");
  SM_ASSERT_GT( Exception, etaPlus, etaMinus, "");
  SM_ASSERT_GT( Exception, initialDelta, 0.0, "");
  SM_ASSERT_GT( Exception, minDelta, 0.0, "");
  SM_ASSERT_GT( Exception, maxDelta, minDelta, "");
  SM_ASSERT_GE( Exception, convergenceGradientNorm, 0.0, "");
  SM_ASSERT_GE( Exception, convergenceDx, 0.0, "");
  SM_ASSERT_GE( Exception, convergenceDObjective, 0.0, "");
  SM_ASSERT_TRUE( Exception, convergenceDx > 0 || convergenceGradientNorm > 0.0 || convergenceDObjective > 0, "");
  SM_ASSERT_GE( Exception, maxIterations, -1, "");
}

std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options)
{
  out << "OptimizerRpropOptions:\n";
  out << "\tetaMinus: " << options.etaMinus << std::endl;
  out << "\tetaPlus: " << options.etaPlus << std::endl;
  out << "\tinitialDelta: " << options.initialDelta << std::endl;
  out << "\tminDelta: " << options.minDelta << std::endl;
  out << "\tmaxDelta: " << options.maxDelta << std::endl;
  out << "\tconvergenceGradientNorm: " << options.convergenceGradientNorm << std::endl;
  out << "\tconvergenceDx: " << options.convergenceDx << std::endl;
  out << "\tconvergenceDObjective: " << options.convergenceDObjective << std::endl;
  out << "\tmaxIterations: " << options.maxIterations << std::endl;
  out << "\tnThreads: " << options.nThreads << std::endl;
  out << "\tmethod: " << options.method << std::endl;
  return out;
}

void RpropReturnValue::reset() {
  convergence = IN_PROGRESS;
  nIterations = nGradEvaluations = nObjectiveEvaluations = 0;
  gradientNorm = std::numeric_limits<double>::signaling_NaN();
  maxDx = std::numeric_limits<double>::signaling_NaN();
  error = std::numeric_limits<double>::max();
  derror = std::numeric_limits<double>::signaling_NaN();
}

bool RpropReturnValue::success() const {
  return convergence != FAILURE && convergence != IN_PROGRESS;
}

bool RpropReturnValue::failure() const {
  return convergence == FAILURE;
}

std::ostream& operator<<(std::ostream& out, const RpropReturnValue::ConvergenceCriterion& convergence) {
  switch (convergence) {
    case RpropReturnValue::ConvergenceCriterion::IN_PROGRESS:
      out << "IN_PROGRESS";
      break;
    case RpropReturnValue::ConvergenceCriterion::FAILURE:
      out << "FAILURE";
      break;
    case RpropReturnValue::ConvergenceCriterion::GRADIENT_NORM:
      out << "GRADIENT_NORM";
      break;
    case RpropReturnValue::ConvergenceCriterion::DX:
      out << "DX";
      break;
    case RpropReturnValue::ConvergenceCriterion::DOBJECTIVE:
      out << "DOBJECTIVE";
      break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const RpropReturnValue& ret) {
  out << "RPropReturnValue: " << std::endl;
  out << "\tconvergence: " << ret.convergence << std::endl;
  out << "\titerations: " << ret.nIterations << std::endl;
  out << "\tgradient norm: " << ret.gradientNorm << std::endl;
  out << "\tobjective: " << ret.error << std::endl;
  out << "\tdobjective: " << ret.derror << std::endl;
  out << "\tmax dx: " << ret.maxDx << std::endl;
  out << "\tevals objective: " << ret.nObjectiveEvaluations << std::endl;
  out << "\tevals gradient: " << ret.nGradEvaluations;
  return out;
}

OptimizerRprop::OptimizerRprop() :
    _options(OptimizerRpropOptions())
{

}

OptimizerRprop::OptimizerRprop(const OptimizerRpropOptions& options) :
    _options(options)
{
  _options.check();
}

OptimizerRprop::OptimizerRprop(const sm::PropertyTree& config) {
  _options = OptimizerRpropOptions(config);
}

OptimizerRprop::~OptimizerRprop()
{
}


void OptimizerRprop::initialize()
{
  ProblemManager::initialize();
  reset();
}

void OptimizerRprop::reset() {
  _dx = ColumnVectorType::Constant(numOptParameters(), 0.0);
  _prev_gradient = ColumnVectorType::Constant(numOptParameters(), 0.0);
  _prev_error = std::numeric_limits<double>::max();
  _delta = ColumnVectorType::Constant(numOptParameters(), _options.initialDelta);
  _returnValue.reset();
}

const RpropReturnValue& OptimizerRprop::optimize()
{
  Timer timeGrad("OptimizerRprop: Compute---Gradient", true);
  Timer timeStep("OptimizerRprop: Compute---Step size", true);
  Timer timeUpdate("OptimizerRprop: Compute---State update", true);

  if (!isInitialized())
    initialize();

  using namespace Eigen;

  std::size_t cnt = 0;
  for (cnt = 0; _options.maxIterations == -1 || cnt < static_cast<size_t>(_options.maxIterations); ++cnt) {

    _returnValue.nIterations++;

    RowVectorType gradient;
    timeGrad.start();
    this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/, false /*TODO: use scaling */, true /*TODO: useDenseJacobianContainer*/);

    // optionally add regularizer
    if (_options.regularizer) {
      JacobianContainerSparse<1> jc(1);
      _options.regularizer->evaluateJacobians(jc);
      SM_FINER_STREAM_NAMED("optimization", "RPROP: Regularization term gradient: " << jc.asDenseMatrix());
      gradient += jc.asDenseMatrix();
    }
    _returnValue.nGradEvaluations++;
    timeGrad.stop();

    SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient " << gradient.format(IOFormat(2, DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite");

    timeStep.start();
    _returnValue.gradientNorm = gradient.norm();

    if (_returnValue.gradientNorm < _options.convergenceGradientNorm) {
      _returnValue.convergence = RpropReturnValue::GRADIENT_NORM;
      SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Current gradient norm " << _returnValue.gradientNorm <<
                            " is smaller than convergenceGradientNorm option -> terminating");
      break;
    }

    // Compute error for iPRop+
    bool errorIncreased = false;
    if (_options.method == OptimizerRpropOptions::IRPROP_PLUS) {
      _returnValue.error = this->evaluateError(_options.nThreads);
      _returnValue.nObjectiveEvaluations++;
      errorIncreased = (_returnValue.error - _prev_error) > 0.0;
      _prev_error = _returnValue.error;
    }

    // determine whether gradient direction switched
    Eigen::Matrix<double, 1, Eigen::Dynamic> gg = _prev_gradient.cwiseProduct(gradient);
    Eigen::Matrix<bool, 1, Eigen::Dynamic> switchNo = gg.array() > 0.0;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> switchYes = gg.array() < 0.0;
    _prev_gradient = gradient;

    for (std::size_t d = 0; d < numOptParameters(); ++d) {

      // Adapt delta
      if (switchNo(d))
        _delta(d) = std::min(_delta(d) * _options.etaPlus, _options.maxDelta);
      else if (switchYes(d))
        _delta(d) = std::max(_delta(d) * _options.etaMinus, _options.minDelta);

      // Note: see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332
      // for a good description of the algorithms
      switch (_options.method) {

        // RPROP_PLUS
        // With backtracking. If gradient switched direction, revert this update.
        case OptimizerRpropOptions::RPROP_PLUS:
        {
          // Compute design variable update vector
          if (switchYes(d)) {
            _dx(d) = -_dx(d); // revert update
            _prev_gradient(d) = 0.0; // this forces switchYes=false in the next step
          } else {
            _dx(d) = -sign(gradient(d))*_delta(d);
          }

          break;
        }

        // RPROP_MINUS
        // No backtracking. Reduce step-length if gradient switched direction,
        // Increase step-length if gradient in same direction.
        case OptimizerRpropOptions::RPROP_MINUS:
        {
          // Compute design variable update vector
          _dx(d) = -sign(gradient(d))*_delta(d);

          break;
        }
        // IRPROP_MINUS
        // In case gradient direction switched, stay at this point for one iteration and
        // then move into the direction of the gradient with half the step-length.
        case OptimizerRpropOptions::IRPROP_MINUS:
        {
          // Compute design variable update vector
          if (switchYes(d))
            _dx(d) = _prev_gradient(d) = 0.0;
          else
            _dx(d) = -sign(gradient(d))*_delta(d);

          break;
        }
        // IRPROP_PLUS
        // Revert only weight updates that have caused changes of the corresponding
        // partial derivatives in case of an error increase.
        case OptimizerRpropOptions::IRPROP_PLUS:
        {

          // Compute design variable update vector
          if (switchYes(d)) {
            if (errorIncreased)
              _dx(d) = -_dx(d); // revert update if gradient direction switched and error increased
            else
              _dx(d) = 0.0;
            _prev_gradient(d) = 0.0; // this forces switchYes=false in the next step
          } else {
            _dx(d) = -sign(gradient(d))*_delta(d);
          }

          break;
        }
      }

    }

    timeStep.stop();

    timeUpdate.start();
    this->applyStateUpdate(_dx);
    timeUpdate.stop();

    _returnValue.maxDx = _dx.cwiseAbs().maxCoeff();
    if (_returnValue.maxDx < _options.convergenceDx) {
      _returnValue.convergence = RpropReturnValue::DX;
      SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Maximum dx coefficient " << _returnValue.maxDx <<
                            " is smaller than convergenceDx option -> terminating");
      break;
    }

    if (_options.method == OptimizerRpropOptions::IRPROP_PLUS) {
      _returnValue.derror = this->evaluateError(_options.nThreads) - _returnValue.error;
      if (fabs(_returnValue.derror) < _options.convergenceDObjective) {
        _returnValue.convergence = RpropReturnValue::DOBJECTIVE;
        SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Change in error " << _returnValue.derror <<
                              " is smaller than convergenceDObjective option -> terminating");
        break;
      }
    }

    SM_FINE_STREAM_NAMED("optimization", _returnValue << std::endl <<
                         "\tdx: " << _dx.transpose() << std::endl <<
                         "\tdelta: " << _delta.transpose());


  }

  SM_DEBUG_STREAM_NAMED("optimization", _returnValue);

  return _returnValue;
}

} // namespace backend
} // namespace aslam
