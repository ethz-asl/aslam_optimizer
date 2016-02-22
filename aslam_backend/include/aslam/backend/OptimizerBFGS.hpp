#ifndef ASLAM_BACKEND_OPTIMIZER_BFGS_HPP
#define ASLAM_BACKEND_OPTIMIZER_BFGS_HPP

#include <aslam/backend/util/ProblemManager.hpp>
#include <aslam/backend/LineSearch.hpp>

namespace sm {
  class PropertyTree;
}

namespace aslam {
  namespace backend {

    struct OptimizerBFGSOptions {
      OptimizerBFGSOptions();
      OptimizerBFGSOptions(const sm::PropertyTree& config);
      double convergenceGradientNorm = 1e-3; /// \brief Stopping criterion on gradient norm
      double convergenceDx = 0.0; /// \brief Stopping criterion on maximum state update coefficient
      double convergenceDObjective = 0.0; /// \brief Stopping criterion on change of objective/error
      int maxIterations = 20; /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria. -1
      std::size_t nThreads = 4; /// \brief The number of threads to use
      LineSearchOptions linesearch;
      boost::shared_ptr<ScalarNonSquaredErrorTerm> regularizer = NULL; /// \brief Regularizer

      void check() const;
    };

    std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerBFGSOptions& options);

    struct BFGSReturnValue {
      enum ConvergenceCriterion { IN_PROGRESS = 0, FAILURE = 1, GRADIENT_NORM = 2, DX = 3, DOBJECTIVE = 4 };
      BFGSReturnValue() { }
      void reset();
      bool success() const;
      bool failure() const;
      ConvergenceCriterion convergence = IN_PROGRESS;
      std::size_t nIterations = 0;
      std::size_t nGradEvaluations = 0;
      std::size_t nObjectiveEvaluations = 0;
      double gradientNorm = std::numeric_limits<double>::signaling_NaN();
      double error = std::numeric_limits<double>::max();
      double derror = std::numeric_limits<double>::signaling_NaN(); /// \brief last change of the error
      double maxDx = std::numeric_limits<double>::signaling_NaN(); /// \brief last maximum change design variables
    };
    std::ostream& operator<<(std::ostream& out, const BFGSReturnValue& ret);
    std::ostream& operator<<(std::ostream& out, const BFGSReturnValue::ConvergenceCriterion& convergence);

    /**
     * \class OptimizerBFGS
     *
     * Broyden-Fletcher-Goldfarb-Shannon algorithm implementation for the ASLAM framework.
     */
    class OptimizerBFGS : private ProblemManager {
    public:

      using ProblemManager::setProblem;
      using ProblemManager::checkProblemSetup;

      typedef boost::shared_ptr<OptimizerBFGS> Ptr;
      typedef boost::shared_ptr<const OptimizerBFGS> ConstPtr;
      typedef OptimizerBFGSOptions Options;

      /// \brief Constructor with default options
      OptimizerBFGS();
      /// \brief Constructor with custom options
      OptimizerBFGS(const Options& options);
      /// \brief Constructor from property tree
      OptimizerBFGS(const sm::PropertyTree& config);
      /// \brief Destructor
      ~OptimizerBFGS();

      /// \brief Run the optimization
      const BFGSReturnValue& optimize();

      /// \brief Return the status
      const BFGSReturnValue& getStatus() const { return _returnValue; }

      /// \brief Get the optimizer options.
      const Options& getOptions() const { return _options; }

      /// \brief Set the optimizer options.
      void setOptions(const Options&);

      /// \brief Return the current gradient norm
      inline double getGradientNorm() { return _returnValue.gradientNorm; }

      /// \brief Get the number of iterations the solver has run.
      ///        If it has never been started, the value will be zero.
      inline std::size_t getNumberOfIterations() const { return _returnValue.nIterations; }

      /// \brief Initialize the optimizer to run on an optimization problem.
      ///        optimize() will call initialize() upon the first call.
      virtual void initialize() override;

      /// \brief Reset internal states but don't re-initialize the whole problem
      void reset();

    private:
      void increaseEvaluateErrorCounter() { _returnValue.nObjectiveEvaluations++; }
      void increaseEvaluateGradientCounter() { _returnValue.nGradEvaluations++; }
      void updateStatus(bool lineSearchSuccess);

    private:

      /// \brief the current set of options
      Options _options;

      /// \brief The current estimate of the inverse Hessian
      Eigen::MatrixXd _Bk;

      /// \brief Line-search class
      LineSearch _linesearch;

      /// \brief Struct returned by optimize method
      BFGSReturnValue _returnValue;

    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_BFGS_HPP */
