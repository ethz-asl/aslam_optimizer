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
    convergenceDx(config.getDouble("convergenceMaxAbsDx", convergenceDx)),
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
  SM_ASSERT_TRUE( Exception, convergenceDx > 0 || convergenceGradientNorm > 0.0, "");
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
  out << "\tmaxIterations: " << options.maxIterations << std::endl;
  out << "\tnThreads: " << options.nThreads << std::endl;
  out << "\tmethod: " << options.method << std::endl;
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


/// \brief initialize the optimizer to run on an optimization problem.
///        This should be called before calling optimize()
void OptimizerRprop::initialize()
{
  ProblemManager::initialize();
  _dx = ColumnVectorType::Constant(numOptParameters(), 0.0);
  _prev_gradient = ColumnVectorType::Constant(numOptParameters(), 0.0);
  _prev_error = std::numeric_limits<double>::max();
  _delta = ColumnVectorType::Constant(numOptParameters(), _options.initialDelta);
  _nIterations = 0;
  _curr_gradient_norm = std::numeric_limits<double>::signaling_NaN();
}

void OptimizerRprop::optimize()
{
  Timer timeGrad("OptimizerRprop: Compute---Gradient", true);
  Timer timeStep("OptimizerRprop: Compute---Step size", true);
  Timer timeUpdate("OptimizerRprop: Compute---State update", true);

  if (!isInitialized())
    initialize();

  if (_options.method == OptimizerRpropOptions::IRPROP_PLUS && std::isnan(_prev_error))
    _prev_error = this->evaluateError(_options.nThreads);

  using namespace Eigen;

  bool isConverged = false;
  for (_nIterations = 0; _options.maxIterations == -1 || _nIterations < static_cast<size_t>(_options.maxIterations); ++_nIterations) {

    RowVectorType gradient;
    timeGrad.start();
    this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);

    // optionally add regularizer
    if (_options.regularizer) {
      JacobianContainer jc(1);
      _options.regularizer->evaluateJacobians(jc);
      SM_FINER_STREAM_NAMED("optimization", "RPROP: Regularization term gradient: " << jc.asDenseMatrix());
      gradient += jc.asDenseMatrix();
    }

    timeGrad.stop();

    SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient " << gradient.format(IOFormat(2, DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite");

    timeStep.start();
    _curr_gradient_norm = gradient.norm();

    if (_curr_gradient_norm < _options.convergenceGradientNorm) {
      isConverged = true;
      SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Current gradient norm " << _curr_gradient_norm <<
                            " is smaller than convergenceGradientNorm option -> terminating");
      break;
    }

    // Compute error for iPRop+
    bool errorIncreased = false;
    if (_options.method == OptimizerRpropOptions::IRPROP_PLUS) {
      const double error = this->evaluateError(_options.nThreads);
      errorIncreased = (error - _prev_error) > 0.0;
      _prev_error = error;
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

    const double maxAbsCoeff = _dx.cwiseAbs().maxCoeff();
    if (maxAbsCoeff < _options.convergenceDx) {
      isConverged = true;
      SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Maximum dx coefficient " << maxAbsCoeff <<
                            " is smaller than convergenceMaxAbsDx option -> terminating");
      break;
    }

    SM_FINE_STREAM_NAMED("optimization", "Number of iterations: " << _nIterations);
    SM_FINE_STREAM_NAMED("optimization", "\t gradient: " << gradient.format(IOFormat(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "[", "]")));
    SM_FINE_STREAM_NAMED("optimization", "\t dx:    " << _dx.format(IOFormat(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "[", "]")) );
    SM_FINE_STREAM_NAMED("optimization", "\t delta:    " << _delta.format(IOFormat(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "[", "]")) );
    SM_FINE_STREAM_NAMED("optimization", "\t norm:     " << _curr_gradient_norm);

    timeStep.stop();

    timeUpdate.start();
    this->applyStateUpdate(_dx);
    timeUpdate.stop();

  }

  std::string convergence = isConverged ? "YES" : "NO";
  SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Convergence " << convergence << " (iterations: " << _nIterations << ", gradient norm: " << _curr_gradient_norm << ")");

}

} // namespace backend
} // namespace aslam
