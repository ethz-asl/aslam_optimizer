#include <aslam/backend/OptimizerSgd.hpp>
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/logging.hpp>

namespace aslam {
namespace backend {

LearningRateScheduleBase::LearningRateScheduleBase(const std::string& name, const double initialRate)
    : _name(name), _initialRate(initialRate)
{
  }

LearningRateScheduleBase::LearningRateScheduleBase(const std::string& name, const sm::PropertyTree& config)
    : _name(name)
{
  _initialRate = config.getDouble("initial", _initialRate);
}

void LearningRateScheduleBase::check() const
{
  SM_ASSERT_GT(Exception, _initialRate, 0.0, "");
}

LearningRateScheduleConstant::LearningRateScheduleConstant(const double rate)
    : LearningRateScheduleBase("constant", rate)
{
  this->check();
}

LearningRateScheduleConstant::LearningRateScheduleConstant(const sm::PropertyTree& config)
    : LearningRateScheduleBase("constant", config)
{
  this->check();
}

double LearningRateScheduleConstant::operator()(const std::size_t /*iteration*/)
{
  return initialRate();
}

LearningRateScheduleOptimal::LearningRateScheduleOptimal(const double initialRate, const double tau)
    : LearningRateScheduleBase("optimal", initialRate), _tau(tau)
{
  this->check();
}

LearningRateScheduleOptimal::LearningRateScheduleOptimal(const sm::PropertyTree& config)
    : LearningRateScheduleBase("optimal", config)
{
  _tau = config.getDouble("tau", _tau);
  this->check();
}

double LearningRateScheduleOptimal::operator()(const std::size_t iteration)
{
  return initialRate() / (1.0 + initialRate() * (static_cast<double>(iteration)/_tau) );
}

void LearningRateScheduleOptimal::check() const
{
  LearningRateScheduleBase::check();
  SM_ASSERT_GT(Exception, _tau, 0.0, "");
}

OptimizerOptionsSgd::OptimizerOptionsSgd()
    : learningRateSchedule(new LearningRateScheduleOptimal(1.0, 0.5))
{
}

OptimizerOptionsSgd::OptimizerOptionsSgd(const sm::PropertyTree& config)
    : OptimizerOptionsBase(config)
{
  useDenseJacobianContainer = config.getDouble("useDenseJacobianContainer", useDenseJacobianContainer);
  const std::string type = config.getString("learning_rate/type");
  if (type == "constant") {
    sm::PropertyTree pt(config, "learning_rate/constant"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleConstant(pt));
  } else if (type == "optimal") {
    sm::PropertyTree pt(config, "learning_rate/optimal"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleOptimal(pt));
  } else {
    SM_THROW(Exception, "Invalid learning rate schedule " << type);
  }
}

void OptimizerOptionsSgd::check() const
{
  OptimizerOptionsBase::check();
  SM_ASSERT_NOTNULL(Exception, learningRateSchedule, "");
  learningRateSchedule->check();
}

std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerOptionsSgd& options)
{
  out << static_cast<OptimizerOptionsBase>(options) << std::endl;
  out << "OptimizerSgdOptions:\n";
  out << "\tuseDenseJacobianContainer: " << (options.useDenseJacobianContainer ? "TRUE" : "FALSE") << std::endl;
  out << "\tlearningRateSchedule: " << (options.learningRateSchedule != nullptr ? options.learningRateSchedule->name() : "N/A");
  return out;
}


void OptimizerStatusSgd::resetImplementation() {
  learningRate = std::numeric_limits<double>::signaling_NaN();
}

std::ostream& operator<<(std::ostream& out, const OptimizerStatusSgd& ret) {
  out << static_cast<OptimizerStatus>(ret) << std::endl;
  out << "OptimizerStatusSgd: " << std::endl;
  out << "\tlearning rate: " << ret.learningRate;
  return out;
}


OptimizerSgd::OptimizerSgd(const Options& options)
    : _options(options)
{
  _options.check();
  this->reset();
}

OptimizerSgd::OptimizerSgd()
    : OptimizerSgd::OptimizerSgd(Options())
{
}


OptimizerSgd::OptimizerSgd(const sm::PropertyTree& config)
    : OptimizerSgd::OptimizerSgd(Options(config))
{
}

OptimizerSgd::~OptimizerSgd()
{
}


void OptimizerSgd::resetImplementation()
{
  _isBatchProcessed = false;
}

void OptimizerSgd::optimizeImplementation()
{
  using namespace Eigen;
  static IOFormat fmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "[", "]");

  SM_ASSERT_FALSE( Exception, _isBatchProcessed, "You should not call run() twice without adding a new batch");

  RowVectorType gradient;

  {
    Timer timeGrad("OptimizerSgd: Compute---Gradient", false);
    problemManager().computeGradient(gradient, _options.numThreadsGradient, false /*TODO: useMEstimator*/, false /* TODO: applyDvScaling*/, _options.useDenseJacobianContainer);
    ++_status.numDerivativeEvaluations;
    _status.gradientNorm = gradient.norm();

    // optionally add regularizer
    if (_options.regularizer) {
      JacobianContainerDense<RowVectorType&, 1> jc(gradient);
      problemManager().addGradientForErrorTerm(jc, _options.regularizer.get(), false /*TODO: useMEstimator*/);
    }
  }

  SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient " << gradient.format(fmt) << " is not finite");

  // Compute learning rate
  _status.learningRate = _options.learningRateSchedule->operator()(_status.numIterations);

  // Update
  RowVectorType dx = -_status.learningRate*1./problemManager().numErrorTerms()*gradient;
  {
    Timer timeUpdate("OptimizerSgd: Compute---State update", true);
    problemManager().applyStateUpdate(dx);
    _status.maxDeltaX = dx.cwiseAbs().maxCoeff();
  }

  SM_FINE_STREAM_NAMED("optimization", _status << std::endl <<
                       "\tgradient: " << gradient.format(fmt) << std::endl <<
                       "\tdx:    " << dx.format(fmt));

  _isBatchProcessed = true;
  _status.numIterations += problemManager().numErrorTerms();
}

} // namespace backend
} // namespace aslam
