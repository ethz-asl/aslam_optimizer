#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/logging.hpp>

namespace aslam {
namespace backend {

OptimizerRpropOptions::OptimizerRpropOptions() :
    etaMinus(0.5),
    etaPlus(1.2),
    initialDelta(0.1), // TODO: what is a good initial value?
    minDelta(1e-20),
    maxDelta(1.0),
    convergenceGradientNorm(1e-3),
    maxIterations(20),
    nThreads(4)
{

}

OptimizerRpropOptions::OptimizerRpropOptions(const sm::PropertyTree& config) :
    etaMinus(config.getDouble("etaMinus", etaMinus)),
    etaPlus(config.getDouble("etaPlus", etaPlus)),
    initialDelta(config.getDouble("initialDelta", initialDelta)),
    minDelta(config.getDouble("minDelta", minDelta)),
    maxDelta(config.getDouble("maxDelta", maxDelta)),
    convergenceGradientNorm(config.getDouble("convergenceGradientNorm", convergenceGradientNorm)),
    maxIterations(config.getInt("maxIterations", maxIterations)),
    nThreads(config.getInt("nThreads", nThreads))
{

}

std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options)
{
  out << "OptimizerRpropOptions:\n";
  out << "\tetaMinus: " << options.etaMinus << std::endl;
  out << "\tetaPlus: " << options.etaPlus << std::endl;
  out << "\tinitialDelta: " << options.initialDelta << std::endl;
  out << "\tminDelta: " << options.minDelta << std::endl;
  out << "\tmaxDelta: " << options.maxDelta << std::endl;
  out << "\tmaxIterations: " << options.maxIterations << std::endl;
  out << "\tnThreads: " << options.nThreads << std::endl;
  return out;
}




OptimizerRprop::OptimizerRprop() :
    _curr_gradient_norm(std::numeric_limits<double>::signaling_NaN()),
    _options(OptimizerRpropOptions()),
    _nIterations(0)
{

}

OptimizerRprop::OptimizerRprop(const OptimizerRpropOptions& options) :
    _curr_gradient_norm(std::numeric_limits<double>::signaling_NaN()),
    _options(options),
    _nIterations(0)
{

}

OptimizerRprop::OptimizerRprop(const sm::PropertyTree& config) :
    _curr_gradient_norm(std::numeric_limits<double>::signaling_NaN()),
    _nIterations(0)
{
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
  _dx.resize(numOptParameters(), 1);
  _prev_gradient.resize(1, numOptParameters());
  _delta = ColumnVectorType::Constant(numOptParameters(), _options.initialDelta);
  _nIterations = 0;
}

void OptimizerRprop::optimize()
{
  Timer timeGrad("OptimizerRprop: Compute---Gradient", true);
  Timer timeStep("OptimizerRprop: Compute---Step size", true);
  Timer timeUpdate("OptimizerRprop: Compute---State update", true);

  if (!isInitialized())
    initialize();

  using namespace Eigen;

  bool isConverged = false;
  for (_nIterations = 0; _options.maxIterations == -1 || _nIterations < static_cast<size_t>(_options.maxIterations); ++_nIterations) {

    RowVectorType gradient;
    timeGrad.start();
    this->computeGradient(gradient, _options.nThreads, false /*TODO: useMEstimator*/);
    timeGrad.stop();

    SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient " << gradient.format(IOFormat(2, DontAlignCols, ", ", ", ", "", "", "[", "]")) << " is not finite");

    timeStep.start();
    _curr_gradient_norm = gradient.norm();

    if (_curr_gradient_norm < _options.convergenceGradientNorm) {
      isConverged = true;
      SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Current gradient norm " << _curr_gradient_norm <<
                            " is smaller than convergenceGradientNorm option -> terminating");
    }

    if (isConverged)
      break;

    _dx.setZero();
    for (std::size_t d = 0; d < numOptParameters(); ++d) {

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
  SM_DEBUG_STREAM_NAMED("optimization", "RPROP: Convergence " << convergence);

}

} // namespace backend
} // namespace aslam
