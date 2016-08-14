#include <algorithm>
#include <aslam/backend/OptimizerSgd.hpp>
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/logging.hpp>

namespace aslam {
namespace backend {

// ******************************* //
// **** LearningRateSchedules **** //
// ******************************* //

LearningRateScheduleBase::LearningRateScheduleBase(const std::string& name)
    : _name(name)
{
}

LearningRateScheduleBase::LearningRateScheduleBase(const std::string& name, const sm::PropertyTree& /*config*/)
    : _name(name)
{
}

void LearningRateScheduleBase::check() const
{
  this->checkImplementation();
}

void LearningRateScheduleBase::initialize(const std::size_t numParameters)
{
  _currentRate.resize(1, numParameters);
  this->initializeImplementation(numParameters);
}

RowVectorType LearningRateScheduleBase::computeDx(const std::size_t nEpoch, const std::size_t nTotal,
                                                  const RowVectorType& gradient)
{
  return this->computeDxImplementation(nEpoch, nTotal, gradient);
}

LearningRateScheduleWithMomentumBase::LearningRateScheduleWithMomentumBase(const std::string& name, const double momentum)
    : LearningRateScheduleBase(name), _momentum(momentum) { }

LearningRateScheduleWithMomentumBase::LearningRateScheduleWithMomentumBase(const std::string& name, const sm::PropertyTree& config)
    : LearningRateScheduleBase(name)
{
  _momentum = config.getDouble("momentum", 0.0);
}

RowVectorType LearningRateScheduleWithMomentumBase::computeDx(const std::size_t nEpoch, const std::size_t nTotal,
                                                              const RowVectorType& gradient)
{
  auto dx = this->computeDxImplementation(nEpoch, nTotal, gradient);
  this->applyMomentum(dx, _prevDx);
  _prevDx = dx;
  return dx;
}

void LearningRateScheduleWithMomentumBase::initialize(const std::size_t numParameters) {
  LearningRateScheduleBase::initialize(numParameters);
  _prevDx.setZero(1, numParameters);
}

void LearningRateScheduleWithMomentumBase::check() const
{
  LearningRateScheduleBase::check();
  SM_ASSERT_NONNEGATIVE(Exception, _momentum, "");
  this->checkImplementation();
}

void LearningRateScheduleWithMomentumBase::applyMomentum(RowVectorType& paramUpdate, const RowVectorType& prevParamUpdate) const
{
  paramUpdate += _momentum * prevParamUpdate;
}

LearningRateScheduleConstant::LearningRateScheduleConstant(const double lr, const double momentum /*= 0.0*/)
    : LearningRateScheduleWithMomentumBase("constant", momentum), _lr(lr)
{
  this->check();
}

LearningRateScheduleConstant::LearningRateScheduleConstant(const sm::PropertyTree& config)
    : LearningRateScheduleWithMomentumBase("constant", config)
{
  _lr = config.getDouble("constant");
  this->check();
}

void LearningRateScheduleConstant::initializeImplementation(const std::size_t numParameters) {
  _currentRate.setConstant(1, numParameters, _lr);
}

RowVectorType LearningRateScheduleConstant::computeDxImplementation(const std::size_t /*nEpoch*/, const std::size_t /*nTotal*/,
                                                                    const RowVectorType& gradient)
{
  SM_ASSERT_EQ(Exception, _currentRate.size(), gradient.size(), "Forgot to call initialize()?");
  return -_currentRate * gradient;
}

void LearningRateScheduleConstant::checkImplementation() const
{
  SM_ASSERT_POSITIVE(Exception, _lr, "");
}

LearningRateScheduleOptimal::LearningRateScheduleOptimal(const double lr, const double tau, const double momentum /*= 0.0*/)
    : LearningRateScheduleWithMomentumBase("optimal", momentum), _lr(lr), _tau(tau)
{
  this->check();
}

LearningRateScheduleOptimal::LearningRateScheduleOptimal(const sm::PropertyTree& config)
    : LearningRateScheduleWithMomentumBase("optimal", config)
{
  _tau = config.getDouble("tau");
  _lr = config.getDouble("initialRate");
  this->check();
}

RowVectorType LearningRateScheduleOptimal::computeDxImplementation(const std::size_t /*nEpoch*/, const std::size_t nTotal,
                                                                   const RowVectorType& gradient)
{
  const double rate = getLr() / (1.0 + getLr() * (static_cast<double>(nTotal)/_tau) );
  _currentRate.setConstant(1, gradient.size(), rate);
  return -rate * gradient;
}

void LearningRateScheduleOptimal::checkImplementation() const
{
  SM_ASSERT_POSITIVE(Exception, _tau, "");
  SM_ASSERT_POSITIVE(Exception, _lr, "");
}

LearningRateScheduleRMSProp::LearningRateScheduleRMSProp(const double lr,
                                                         const double rho /*= 0.9*/,
                                                         const double epsilon /*= 1e-6*/,
                                                         const double momentum /*= 0.0*/,
                                                         const bool isStepAdapt /*= false*/,
                                                         const double stepFactor /*= 0.2*/,
                                                         const double minLr /*= 0.*/,
                                                         const double maxLr /*= std::numeric_limits<double>::infinity()*/)
    : LearningRateScheduleWithMomentumBase("RMSprop", momentum), _lr(lr), _rho(rho), _epsilon(epsilon),
      _isStepAdapt(isStepAdapt), _stepFactor(stepFactor), _minLr(minLr), _maxLr(maxLr)
{
  this->check();
}

LearningRateScheduleRMSProp::LearningRateScheduleRMSProp(const sm::PropertyTree& config)
    : LearningRateScheduleWithMomentumBase("RMSprop", config)
{
  _lr = config.getDouble("lr");
  _rho = config.getDouble("rho", _rho);
  _epsilon = config.getDouble("epsilon", _epsilon);
  _isStepAdapt = config.getBool("step_adapt", _isStepAdapt);
  _stepFactor = config.getDouble("step_factor", _stepFactor);
  _minLr = config.getDouble("min_lr", _minLr);
  _maxLr = config.getDouble("max_lr", _maxLr);
  this->check();
}

void LearningRateScheduleRMSProp::checkImplementation() const
{
  SM_ASSERT_POSITIVE(Exception, _lr, "");
  SM_ASSERT_GE_LT(Exception, _rho, 0.0, 1.0, "");
  SM_ASSERT_POSITIVE(Exception, _epsilon, "");
  SM_ASSERT_NONNEGATIVE(Exception, _stepFactor, "");
  SM_ASSERT_NONNEGATIVE(Exception, _minLr, "");
  SM_ASSERT_POSITIVE(Exception, _maxLr, "");
}

void LearningRateScheduleRMSProp::initializeImplementation(const std::size_t numParameters)
{
  _gradSqrAverage.setOnes(1, numParameters);
  _alpha.setConstant(1, numParameters, _lr);
}

RowVectorType LearningRateScheduleRMSProp::computeDxImplementation(const std::size_t /*iteration*/, const std::size_t /*nTotal*/,
                                                                   const RowVectorType& gradient)
{
  SM_ASSERT_EQ(Exception, _gradSqrAverage.size(), gradient.size(), "");
  _gradSqrAverage = _rho * _gradSqrAverage + (1. - _rho) * gradient.cwiseProduct(gradient);
  const RowVectorType rmsGrad = (_gradSqrAverage.array() + _epsilon).cwiseSqrt();
  _currentRate = _alpha.cwiseQuotient(rmsGrad);
  const RowVectorType dx = -_currentRate.cwiseProduct(gradient);

  // Adapt learning rate based on this and previous update direction
  if (_isStepAdapt) {
    const RowVectorType sgn = dx.cwiseProduct(getPrevDx());
    for (int i=0; i<sgn.size(); ++i) {
      if (sgn(i) > 0 ) {
        _alpha(i) *= (1. + _stepFactor); // increase if update kept sign
      } else if (sgn(i) > 0 ) {
        _alpha(i) *= (1. - _stepFactor); // decrease if update changed sign
      }
    }
    _alpha = _alpha.cwiseMax(_minLr).cwiseMin(_maxLr);
  }

  return dx;
}

LearningRateScheduleAdaDelta::LearningRateScheduleAdaDelta(const double rho /*= 0.9*/, const double epsilon /*= 1e-6*/, const double lr /*= 1.0*/)
    : LearningRateScheduleBase("Adadelta"), _lr(lr), _rho(rho), _epsilon(epsilon)
{
  this->check();
}

LearningRateScheduleAdaDelta::LearningRateScheduleAdaDelta(const sm::PropertyTree& config)
    : LearningRateScheduleBase("Adadelta", config)
{
  _lr = config.getDouble("lr", _lr);
  _rho = config.getDouble("rho", _rho);
  _epsilon = config.getDouble("eps", _epsilon);
  this->check();
}

void LearningRateScheduleAdaDelta::initializeImplementation(const std::size_t numParameters)
{
  _gradSqrAverage.setZero(1, numParameters);
  _dxSqrAverage.setZero(1, numParameters);
}

RowVectorType LearningRateScheduleAdaDelta::computeDxImplementation(const std::size_t /*iteration*/, const std::size_t /*nTotal*/,
                                                                    const RowVectorType& gradient)
{
  SM_ASSERT_EQ(Exception, _gradSqrAverage.size(), gradient.size(), "");
  SM_ASSERT_EQ(Exception, _dxSqrAverage.size(), gradient.size(), "");

  _gradSqrAverage = _rho * _gradSqrAverage + (1. - _rho) * gradient.cwiseProduct(gradient);
  const RowVectorType rmsGrad = (_gradSqrAverage.array() + _epsilon).cwiseSqrt();
  const RowVectorType rmsDx = (_dxSqrAverage.array() + _epsilon).cwiseSqrt();
  _currentRate = _lr * rmsDx.cwiseQuotient(rmsGrad);
  const RowVectorType dx =  -_currentRate.cwiseProduct(gradient);
  _dxSqrAverage = _rho * _dxSqrAverage + (1. - _rho) * dx.cwiseProduct(dx);
  return dx;
}

void LearningRateScheduleAdaDelta::checkImplementation() const
{
  SM_ASSERT_POSITIVE(Exception, _lr, "");
  SM_ASSERT_GE_LT(Exception, _rho, 0.0, 1.0, "");
  SM_ASSERT_POSITIVE(Exception, _epsilon, "");
}

LearningRateScheduleAdam::LearningRateScheduleAdam(const double lr, const double rho1 /*= 0.9*/,
                                                   const double rho2 /*= 0.999*/, const double epsilon /*= 1e-8*/)
    : LearningRateScheduleBase("Adam"), _lr(lr), _rho1(rho1), _rho2(rho2), _epsilon(epsilon), _beta1(rho1), _beta2(rho2)
{
  this->check();
}

LearningRateScheduleAdam::LearningRateScheduleAdam(const sm::PropertyTree& config)
    : LearningRateScheduleBase("Adam", config)
{
  _lr = config.getDouble("lr");
  _rho1 = config.getDouble("rho1", _rho1);
  _rho2 = config.getDouble("rho2", _rho2);
  _epsilon = config.getDouble("eps", _epsilon);
  _beta1 = _rho1;
  _beta2 = _rho2;
  this->check();
}

void LearningRateScheduleAdam::initializeImplementation(const std::size_t numParameters)
{
  _gradAverage.setZero(1, numParameters);
  _gradSqrAverage.setZero(1, numParameters);
  _beta1 = _rho1;
  _beta2 = _rho2;
}

RowVectorType LearningRateScheduleAdam::computeDxImplementation(const std::size_t /*iteration*/, const std::size_t /*nTotal*/,
                                                                const RowVectorType& gradient)
{
  SM_ASSERT_EQ(Exception, _gradAverage.size(), gradient.size(), "");
  SM_ASSERT_EQ(Exception, _gradSqrAverage.size(), gradient.size(), "");

  _gradAverage = _rho1 * _gradAverage + (1. - _rho1) * gradient;
  _gradSqrAverage = _rho2 * _gradSqrAverage + (1. - _rho2) * gradient.cwiseProduct(gradient);

  _currentRate = _lr * sqrt(1. - _beta2)/sqrt(1. - _beta1) * ( _gradSqrAverage.cwiseSqrt().array() + _epsilon ).cwiseInverse();
  _beta1 *= _rho1;
  _beta2 *= _rho2;

  return -_currentRate.cwiseProduct(_gradAverage);
}

void LearningRateScheduleAdam::checkImplementation() const
{
  SM_ASSERT_POSITIVE(Exception, _lr, "");
  SM_ASSERT_GE_LT(Exception, _rho1, 0.0, 1.0, "");
  SM_ASSERT_GE_LT(Exception, _rho2, 0.0, 1.0, "");
  SM_ASSERT_POSITIVE(Exception, _epsilon, "");
}


// ******************************* //
// ***** OptimizerOptionsSgd ***** //
// ******************************* //

OptimizerOptionsSgd::OptimizerOptionsSgd()
    : learningRateSchedule(new LearningRateScheduleOptimal(1.0, 0.5))
{
}

OptimizerOptionsSgd::OptimizerOptionsSgd(const sm::PropertyTree& config)
    : OptimizerOptionsBase(config)
{
  batchSize = config.getInt("batchSize", batchSize);
  useDenseJacobianContainer = config.getBool("useDenseJacobianContainer", useDenseJacobianContainer);
  std::string type = config.getString("learning_rate/type");
  std::transform(type.begin(), type.end(), type.begin(), ::tolower);
  if (type == "constant") {
    sm::PropertyTree pt(config, "learning_rate/constant"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleConstant(pt));
  } else if (type == "optimal") {
    sm::PropertyTree pt(config, "learning_rate/optimal"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleOptimal(pt));
  } else if (type == "adadelta") {
    sm::PropertyTree pt(config, "learning_rate/adadelta"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleAdaDelta(pt));
  } else if (type == "rmsprop") {
    sm::PropertyTree pt(config, "learning_rate/rmsprop"); // extract subtree
    learningRateSchedule.reset(new LearningRateScheduleRMSProp(pt));
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
  out << "\tbatchSize: " << options.batchSize << std::endl;
  out << "\tuseDenseJacobianContainer: " << (options.useDenseJacobianContainer ? "TRUE" : "FALSE") << std::endl;
  out << "\tlearningRateSchedule: " << (options.learningRateSchedule != nullptr ? options.learningRateSchedule->getName() : "N/A");
  return out;
}

// ******************************** //
// ****** OptimizerStatusSgd ****** //
// ******************************** //

void OptimizerStatusSgd::resetImplementation() {
  numBatches = numSubIterations = numTotalIterations = 0;
  processedErrorTerms.clear();
}

std::ostream& operator<<(std::ostream& out, const OptimizerStatusSgd& ret) {
  out << static_cast<OptimizerStatus>(ret) << std::endl;
  out << "OptimizerStatusSgd: " << std::endl;
  out << "\tnumber of batches: " << ret.numBatches << std::endl;
  out << "\tnumber of sub-iterations: " << ret.numSubIterations << std::endl;
  out << "\ttotal number of iterations: " << ret.numTotalIterations << std::endl;
  out << "\tprocessed error terms: [";
  std::string delim = "";
  for (const auto& index : ret.processedErrorTerms) {
    out << delim << index;
    delim = ", ";
  }
  out << "]";
  return out;
}

// ****************************** //
// ******** OptimizerSgd ******** //
// ****************************** //

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
}

void OptimizerSgd::initializeImplementation()
{
  OptimizerProblemManagerBase::initializeImplementation();
  _options.learningRateSchedule->initialize(problemManager().numOptParameters());
}

void OptimizerSgd::setCounters(const std::size_t numEpochs, const std::size_t numTotal)
{
  _status.numIterations = numEpochs;
  _status.numTotalIterations = numTotal;
}

struct Rng {
  static Rng& instance() { static Rng random; return random; }
  std::mt19937& generator() { return _gen; }
 private:
  std::random_device _rd;
  std::mt19937 _gen;
  Rng() : _gen(_rd()) { }
};

void OptimizerSgd::setRandomSeed(std::size_t seed) const {
  Rng::instance().generator().seed(seed);
}

void OptimizerSgd::optimizeImplementation()
{
  using namespace std;
  using namespace Eigen;
  static IOFormat fmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "[", "]");

  RowVectorType gradient;

  for ( ; _options.maxIterations == -1 || _status.numIterations < static_cast<std::size_t>(_options.maxIterations); ++_status.numIterations) { // epochs/passes over the data

    _callbackManager.issueCallback( {callback::Occasion::ITERATION_START} );

    // partition the error terms into mini-batches and shuffle them
    _status.numBatches = std::ceil(static_cast<double>(problemManager().numErrorTerms())/_options.batchSize);
    vector< std::vector<size_t> > batches(_status.numBatches);
    size_t cnt = 0;
    for (auto& batch : batches) {
      for (size_t n = 0; n < _options.batchSize && cnt < problemManager().numErrorTerms(); ++n, ++cnt) {
        batch.push_back(cnt);
      }
    }

    std::shuffle(batches.begin(), batches.end(), Rng::instance().generator());
    SM_VERBOSE_STREAM_NAMED("optimization", "OptimizerSgd: Splitted " << problemManager().numErrorTerms() <<
                            " error terms in " <<  _status.numBatches << " batches");

    // iterate over the mini-batches
    for (_status.numSubIterations = 0; _status.numSubIterations < batches.size(); ++_status.numSubIterations, ++_status.numTotalIterations) {

      _status.processedErrorTerms = batches[_status.numSubIterations];

      SM_DEBUG_STREAM_NAMED("optimization", "OptimizerSgd: Processing batch number " << _status.numSubIterations + 1 << "/" << batches.size() <<
                            " with error terms in interval [" << _status.processedErrorTerms.front() << ", " << _status.processedErrorTerms.back() + 1 << ").");

      {
        Timer timeGrad("OptimizerSgd: Compute---Gradient", false);
        gradient = RowVectorType::Zero(1, problemManager().numOptParameters());
        problemManager().computeGradient(gradient, _status.processedErrorTerms.front(), _status.processedErrorTerms.back() + 1, _options.numThreadsGradient,
                                         false /*TODO: useMEstimator*/, false /* TODO: applyDvScaling*/, _options.useDenseJacobianContainer);
        ++_status.numDerivativeEvaluations;
        _status.gradientNorm = gradient.norm();

        // optionally add regularizer
        if (_options.regularizer) {
          JacobianContainerSparse<1> jc(1);
          _options.regularizer->evaluateJacobians(jc);
          SM_FINER_STREAM_NAMED("optimization", "OptimizerSgd: Gradient of regularization term: " << endl << jc.asDenseMatrix());
          gradient += jc.asDenseMatrix()/_status.numBatches;
        }
      }

      SM_ASSERT_TRUE_DBG(Exception, gradient.allFinite (), "Gradient " << gradient.format(fmt) << " is not finite");

      // Update
      const auto dx = _options.learningRateSchedule->computeDx(_status.numIterations, _status.numTotalIterations, gradient);

      _callbackManager.issueCallback( {callback::Occasion::DESIGN_VARIABLE_UPDATE_COMPUTED} );
      {
        Timer timeUpdate("OptimizerSgd: Compute---State update", true);
        problemManager().applyStateUpdate(dx);
        _status.maxDeltaX = dx.cwiseAbs().maxCoeff();
      }
      _callbackManager.issueCallback( {callback::Occasion::DESIGN_VARIABLES_UPDATED} );

      SM_DEBUG_STREAM_NAMED("optimization.learningrate", "OptimizerSgd: Learning rate" << std::endl <<
                            "\tgradient: " << gradient.format(fmt) << std::endl <<
                            "\tlearning rate: " << _options.learningRateSchedule->getCurrentRate().format(fmt) << std::endl <<
                            "\tdx:    " << dx.format(fmt));
      SM_FINE_STREAM_NAMED("optimization", _status << std::endl <<
                           "\tgradient: " << gradient.format(fmt) << std::endl <<
                           "\tlearning rate: " << _options.learningRateSchedule->getCurrentRate().format(fmt) << std::endl <<
                           "\tdx:    " << dx.format(fmt));

    }

    if (!_status.failure())
      SM_DEBUG_STREAM_NAMED("optimization", _status);
    else
      SM_ERROR_STREAM(_status);

    _callbackManager.issueCallback( {callback::Occasion::ITERATION_END} );
  }
}

} // namespace backend
} // namespace aslam
