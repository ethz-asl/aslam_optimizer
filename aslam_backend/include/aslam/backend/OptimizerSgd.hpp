#ifndef ASLAM_BACKEND_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_HPP
#define ASLAM_BACKEND_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <boost/shared_ptr.hpp>

#include <sm/PropertyTree.hpp>

#include <aslam/backend/util/ProblemManager.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/util/OptimizerProblemManagerBase.hpp>

namespace sm {
class PropertyTree;
}

namespace aslam {
namespace backend {

/**
 * \class LearningRateScheduleBase
 * Base class for learning rate schedules
 */
class LearningRateScheduleBase {
 public:
  /// \brief Constructor
  LearningRateScheduleBase(const std::string& name);

  /// \brief Constructor from property tree
  LearningRateScheduleBase(const std::string& name, const sm::PropertyTree& config);

  /// \brief Destructor
  virtual ~LearningRateScheduleBase() { }

  /**
   * Returns the learning rate
   * @param nEpoch Number of full passes through data
   * @param gradient Total number of steps performed so far
   * @return Suggested design variable update
   */
  virtual RowVectorType computeDx(const std::size_t nEpoch, const std::size_t nTotal,
                                  const RowVectorType& gradient);

  /// \brief Initialize for a problem with \p numParameters parameters
  virtual void initialize(const std::size_t /*numParameters*/);

  /// \brief Check for consistency
  virtual void check() const;

  const RowVectorType& getCurrentRate() const {  return _currentRate; }
  const std::string& getName() const { return _name; }

 protected:
  /// \brief Implementation of the consistency check for derived schedules
  virtual void checkImplementation() const { }

  /// \brief Implementation of the initialize method for derived schedules
  virtual void initializeImplementation(const std::size_t /*numParameters*/) { }

  /// \brief Implementation of the initialize method for derived schedules
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) = 0;

 protected:
  RowVectorType _currentRate;  /// \brief The latest learning rate per design variable

 private:
  std::string _name;    /// \brief Name of the schedule
};
std::ostream& operator<<(std::ostream& os, const LearningRateScheduleBase& schedule);

/**
 * \class LearningRateScheduleWithMomentumBase
 * Base class for learning rate schedules with momentum term
 */
class LearningRateScheduleWithMomentumBase : public LearningRateScheduleBase {
 public:
  LearningRateScheduleWithMomentumBase(const std::string& name, const double momentum);
  LearningRateScheduleWithMomentumBase(const std::string& name, const sm::PropertyTree& config);
  virtual void check() const override;
  virtual RowVectorType computeDx(const std::size_t nEpoch, const std::size_t nTotal,
                                  const RowVectorType& gradient) override;
  virtual void initialize(const std::size_t numParameters) override;

  double getMomentum() const { return _momentum; }
  void setMomentum(double momentum) { _momentum = momentum; }

  const RowVectorType& getPrevDx() const { return _prevDx; }

 private:
  void applyMomentum(RowVectorType& paramUpdate, const RowVectorType& prevParamUpdate) const;

 private:
  double _momentum; /// \brief Momentum factor >= 0.0
  RowVectorType _prevDx;  /// \brief Previous parameter update vector
};

/**
 * \class LearningRateScheduleConstant
 * Fixed learning rate
 */
class LearningRateScheduleConstant : public LearningRateScheduleWithMomentumBase {
 public:
  LearningRateScheduleConstant(const double constantRate, const double momentum = 0.0);
  LearningRateScheduleConstant(const sm::PropertyTree& config);

  double getLr() const { return _lr; }
  void setLr(double lr) { _lr = lr; }

 private:
  virtual void checkImplementation() const override;
  virtual void initializeImplementation(const std::size_t numParameters) override;
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) override;
 private:
  double _lr;   /// \brief Constant learning rate, tune freely (> 0)
};

/**
 * \class LearningRateScheduleOptimal
 * As in http://scikit-learn.org/stable/modules/sgd.html#id1
 */
class LearningRateScheduleOptimal : public LearningRateScheduleWithMomentumBase {
 public:
  LearningRateScheduleOptimal(const double lr, const double tau, const double momentum = 0.0);
  LearningRateScheduleOptimal(const sm::PropertyTree& config);

  double getLr() const { return _lr; }
  void setLr(double lr) { _lr = lr; }

  double getTau() const {  return _tau; }
  void setTau(double tau) { _tau = tau; }

 private:
  virtual void checkImplementation() const override;
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) override;
 private:
  double _lr;   /// \brief Initial learning rate, tune freely (> 0)
  double _tau;  /// \brief Decay time constant, the larger the slower (> 0)
};

/**
 * \class LearningRateScheduleRMSProp
 * As in http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
 */
class LearningRateScheduleRMSProp : public LearningRateScheduleWithMomentumBase {
 public:
  LearningRateScheduleRMSProp(const double lr, const double rho = 0.9,
                              const double epsilon = 1e-6, const double momentum = 0.0,
                              const bool isStepAdapt = false, const double stepFactor = 0.2,
                              const double minLr = 0., const double maxLr = std::numeric_limits<double>::infinity());
  LearningRateScheduleRMSProp(const sm::PropertyTree& config);

  double getLr() const { return _lr; }
  void setLr(double lr) { _lr = lr; }

  double getRho() const { return _rho; }
  void setRho(double rho) { _rho = rho; }

  double getEpsilon() const { return _epsilon; }
  void setEpsilon(double epsilon) { _epsilon = epsilon; }

  bool isIsStepAdapt() const { return _isStepAdapt; }
  void setIsStepAdapt(bool isStepAdapt = true) { _isStepAdapt = isStepAdapt; }

  double getMaxLr() const { return _maxLr; }
  void setMaxLr(double maxLr = std::numeric_limits<double>::infinity()) { _maxLr = maxLr; }

  double getMinLr() const { return _minLr; }
  void setMinLr(double minLr = 0.) { _minLr = minLr; }

  double getStepFactor() const { return _stepFactor; }
  void setStepFactor(double stepFactor = 0.2) { _stepFactor = stepFactor; }


  const RowVectorType& getGradSqrAverage() const { return _gradSqrAverage; }
  const RowVectorType& getAlpha() const { return _alpha; }

 private:
  virtual void checkImplementation() const override;
  virtual void initializeImplementation(const std::size_t numParameters) override;
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) override;
 private:
  double _lr;                     /// \brief Learning rate, tune freely (> 0)
  double _rho = 0.9;              /// \brief Smoothing factor of squared gradient running average, the larger the more smoothing (0 <= rho < 1)
  double _epsilon = 1e-6;         /// \brief Fuzz factor for non-zero denominators (> 0)
  bool _isStepAdapt = true;       /// \brief Whether or not to adapt the learning rate per parameter
  double _stepFactor = 0.2;       /// \brief Only relevant when \p _isStepAdapt is true, will update learning rate with (1 +/- _stepFactor) depending on whether update changed sign
  double _minLr = 0.;             /// \brief Minimum learning rate
  double _maxLr = std::numeric_limits<double>::infinity(); /// \brief Maximum learning rate

  RowVectorType _gradSqrAverage;  /// \brief Squared gradient running average
  RowVectorType _alpha;           /// \brief On-the-fly per parameter learning rate
};

/**
 * \class LearningRateScheduleAdaDelta
 * As in https://arxiv.org/abs/1212.5701
 */
class LearningRateScheduleAdaDelta : public LearningRateScheduleBase {
 public:
  LearningRateScheduleAdaDelta(const double rho = 0.9, const double epsilon = 1e-6, const double lr = 1.0);
  LearningRateScheduleAdaDelta(const sm::PropertyTree& config);

  double getLr() const { return _lr; }
  void setLr(double lr) { _lr = lr; }

  double getRho() const { return _rho; }
  void setRho(double rho) { _rho = rho; }

  const RowVectorType& getDxSqrAverage() const { return _dxSqrAverage; }
  const RowVectorType& getGradSqrAverage() const { return _gradSqrAverage; }

  double getEpsilon() const { return _epsilon; }
  void setEpsilon(double epsilon) { _epsilon = epsilon; }

 private:
  virtual void checkImplementation() const override;
  virtual void initializeImplementation(const std::size_t numParameters) override;
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) override;
 private:
  double _lr = 1.0;               /// \brief Learning rate, usually leave at default value (> 0)
  double _rho = 0.9;              /// \brief Smoothing factor for running averages, the larger the more smoothing (0 <= rho < 1)
  double _epsilon = 1e-6;         /// \brief Fuzz factor for non-zero denominators (> 0)
  RowVectorType _gradSqrAverage;  /// \brief Squared gradient running average
  RowVectorType _dxSqrAverage;    /// \brief Squared parameter update running average
};

/**
 * \class LearningRateScheduleAdam
 * As in http://caffe.berkeleyvision.org/tutorial/solver.html
 */
class LearningRateScheduleAdam : public LearningRateScheduleBase {
 public:
  LearningRateScheduleAdam(const double lr, const double rho1 = 0.9,
                           const double rho2 = 0.999, const double epsilon = 1e-8);
  LearningRateScheduleAdam(const sm::PropertyTree& config);

  double getLr() const { return _lr; }
  void setLr(double lr) { _lr = lr; }

  double getEpsilon() const { return _epsilon; }
  void setEpsilon(double eps) { _epsilon = eps; }

  double getRho1() const { return _rho1; }
  double getRho2() const { return _rho2; }

  const RowVectorType& getGradAverage() const { return _gradAverage; }
  const RowVectorType& getGradSqrAverage() const { return _gradSqrAverage; }

 private:
  virtual void checkImplementation() const override;
  virtual void initializeImplementation(const std::size_t numParameters) override;
  virtual RowVectorType computeDxImplementation(const std::size_t nEpoch, const std::size_t nTotal,
                                                const RowVectorType& gradient) override;
 private:
  double _lr;                     /// \brief Learning rate
  double _rho1 = 0.9;             /// \brief Smoothing factor for gradient running averages, the larger the more smoothing
  double _rho2 = 0.999;           /// \brief Smoothing factor for squared gradient running averages, the larger the more smoothing
  double _epsilon = 1e-8;         /// \brief Fuzz factor for non-zero denominators

  double _beta1, _beta2;          /// \brief _rho1^t, _rho2^t
  RowVectorType _gradAverage;     /// \brief Gradient running average
  RowVectorType _gradSqrAverage;  /// \brief Squared gradient running average
};


/// \struct OptimizerOptionsSgd
struct OptimizerOptionsSgd : public OptimizerOptionsBase {
  OptimizerOptionsSgd();
  OptimizerOptionsSgd(const sm::PropertyTree& config);
  std::size_t batchSize = 1; /// \brief Mini-batch size
  bool useDenseJacobianContainer = true; /// \brief Whether or not to use a dense Jacobian container
  boost::shared_ptr<LearningRateScheduleBase> learningRateSchedule;
  boost::shared_ptr<ScalarNonSquaredErrorTerm> regularizer = NULL; /// \brief Regularizer

  void check() const override;
};
std::ostream& operator<<(std::ostream& out, const OptimizerOptionsSgd& options);

/// \struct OptimizerStatusSgd
struct OptimizerStatusSgd : public OptimizerStatus {
  OptimizerStatusSgd() { }
  std::size_t numBatches = 0; /// \brief Number of batches to process
  std::size_t numSubIterations = 0; /// \brief Number of sub-iterations (one batch) within epoch
  std::size_t numTotalIterations = 0; /// \brief Total count of sub-iterations
  std::vector<std::size_t> processedErrorTerms; /// \brief The currently processed or just finished error terms
 private:
  void resetImplementation() override;
};
std::ostream& operator<<(std::ostream& out, const OptimizerStatusSgd& ret);

/**
 * \class OptimizerSgd
 *
 * Stochastic gradient descent implementation for the ASLAM framework.
 */
class OptimizerSgd : public OptimizerProblemManagerBase {

 public:

  typedef boost::shared_ptr<OptimizerSgd> Ptr;
  typedef boost::shared_ptr<const OptimizerSgd> ConstPtr;
  typedef OptimizerOptionsSgd Options;
  typedef OptimizerStatusSgd Status;

  /// \brief Constructor with default options
  OptimizerSgd();
  /// \brief Constructor with custom options
  OptimizerSgd(const Options& options);
  /// \brief Constructor from property tree
  OptimizerSgd(const sm::PropertyTree& config);
  /// \brief Destructor
  ~OptimizerSgd();

  /// \brief Add a batch of error terms. Use this method if you cannot keep all error terms in memory.
  /// Otherwise, just use setProblem(problem), where problem contains all error terms.
  /// They will be processed in mini-batches of configurable size then.
  template <typename Container>
  inline void addBatch(const Container& ets);

  /// \brief Return the status
  const Status& getStatus() const override { return _status; }

  /// \brief Const getter for the optimizer options.
  const Options& getOptions() const override { return _options; }

  /// \brief Mutable getter for the optimizer options (we explicitly allow direct modification of options).
  Options& getOptions() { return _options; }

  /// \brief Set the optimizer options.
  void setOptions(const Options& options) { _options = options; }

  /// \brief Set the optimizer options.
  void setOptions(const OptimizerOptionsBase& options) override { static_cast<OptimizerOptionsBase&>(_options) = options; }

  /// \brief Manually set the counters. Use this, if you have
  ///        already incorporated some training samples in another way
  ///        (e.g. via batch optimization) in the beginning.
  void setCounters(const std::size_t numEpochs, const std::size_t numTotal);

  /// \brief Set seed for random number generator used to shuffle data
  void setRandomSeed(std::size_t seed) const;

 private:

  /// \brief Incorporate the information of the previously added error terms.
  void optimizeImplementation() override;

  /// \brief Reset the learning rate of the optimizer. The previously added batch will still be stored,
  ///        but labeled as not processed.
  void resetImplementation() override;

  void initializeImplementation() override;

  /// \brief Given the current gradient, compute the state update vector
//  virtual void computeStateUpdate(RowVectorType& dx, const RowVectorType& gradient);

  template <typename T>
  void addErrorTerm(boost::shared_ptr<OptimizationProblem> problem, T* et) const;
  template <typename T>
  void addErrorTerm(boost::shared_ptr<OptimizationProblem> problem, boost::shared_ptr<T> et) const;

  /// \brief Extracts a problem with the error terms in range [start, end)
  void computeGradient(RowVectorType& gradient, const std::size_t start, const std::size_t end) const;

 private:

  /// \brief the current set of options
  Options _options;

  /// \brief State information of the optimizer
  Status _status;

};





template <typename T>
void OptimizerSgd::addErrorTerm(boost::shared_ptr<OptimizationProblem> problem, T* et) const {
  problem->addErrorTerm(et, false);
}

template <typename T>
void OptimizerSgd::addErrorTerm(boost::shared_ptr<OptimizationProblem> problem, boost::shared_ptr<T> et) const {
  problem->addErrorTerm(et);
}

/// \brief Add a batch of error terms
template <typename Container>
void OptimizerSgd::addBatch(const Container& ets) {
  Timer timer("OptimizerSgd: Compute---Add batch", false);
  boost::shared_ptr<OptimizationProblem> problem(new OptimizationProblem());
  DesignVariable::set_t dvs;
  for (auto& et : ets) {
    et->getDesignVariables(dvs);
    for (auto& dv : dvs)
      problem->addDesignVariable(dv, false);
    addErrorTerm(problem, et);
  }
  problemManager().setProblem(problem);
}

} // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
