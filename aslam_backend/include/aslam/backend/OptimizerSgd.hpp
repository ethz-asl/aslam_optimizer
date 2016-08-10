#ifndef ASLAM_BACKEND_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_HPP
#define ASLAM_BACKEND_OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <iostream>
#include <string>
#include <vector>

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
  LearningRateScheduleBase(const std::string& name, const double initialRate);
  LearningRateScheduleBase(const std::string& name, const sm::PropertyTree& config);
  virtual ~LearningRateScheduleBase() { }
  /// \brief Return the learning rate for iteration \p iteration
  virtual double operator()(const std::size_t iteration) = 0;
  virtual void check() const;
  const std::string& name() const { return _name; }
  double initialRate() const { return _initialRate; }
 private:
  std::string _name;         /// \brief Name of the schedule
  double _initialRate = 1.0; /// \brief Initial learning rate
};
std::ostream& operator<<(std::ostream& os, const LearningRateScheduleBase& schedule);

/**
 * \class LearningRateScheduleConstant
 * Fixed learning rate
 */
class LearningRateScheduleConstant : public LearningRateScheduleBase {
 public:
  LearningRateScheduleConstant(const double rate);
  LearningRateScheduleConstant(const sm::PropertyTree& config);
  virtual double operator()(const std::size_t /*iteration*/) override;
};

/**
 * \class LearningRateScheduleOptimal
 * As in http://scikit-learn.org/stable/modules/sgd.html#id1
 */
class LearningRateScheduleOptimal : public LearningRateScheduleBase {
 public:
  LearningRateScheduleOptimal(const double initialRate, const double tau);
  LearningRateScheduleOptimal(const sm::PropertyTree& config);
  virtual double operator()(const std::size_t iteration) override;
  virtual void check() const override;
  double tau() const { return _tau; }
 private:
  double _tau = 1.0;
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
  double learningRate = std::numeric_limits<double>::signaling_NaN();
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

  /// \brief Add a batch of error terms
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

  /// \brief Manually increment the number of iterations. Use this, if you have
  ///        already incorporated some training samples in another way
  ///        (e.g. via batch optimization) in the beginning.
  void incrementNumberOfIterations(std::size_t n) { _status.numIterations += n; }

 private:

  /// \brief Incorporate the information of the previously added batch of error terms
  void optimizeImplementation() override;

  /// \brief Reset the learning rate of the optimizer. The previously added batch will still be stored,
  ///        but labeled as not processed.
  void resetImplementation() override;

  /// \brief Given the current gradient, compute the state update vector
  virtual void computeStateUpdate(RowVectorType& dx, const RowVectorType& gradient);

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

  /// \brief Was the last batch processed?
  bool _isBatchProcessed = false;

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
  problemManager().initialize();
  _isBatchProcessed = false;
}

} // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
