#ifndef ASLAM_BACKEND_OPTIMIZER_RPROP_HPP
#define ASLAM_BACKEND_OPTIMIZER_RPROP_HPP

#include <aslam/backend/util/ProblemManager.hpp>

namespace sm {
  class PropertyTree;
}

namespace aslam {
  namespace backend {

    struct OptimizerRpropOptions {

      enum Method { RPROP_PLUS, RPROP_MINUS, IRPROP_MINUS, IRPROP_PLUS };

      OptimizerRpropOptions();
      OptimizerRpropOptions(const sm::PropertyTree& config);
      double etaMinus = 0.5; /// \brief Decrease factor for step size if gradient direction changes
      double etaPlus = 1.2; /// \brief Increase factor for step size if gradient direction is same
      double initialDelta = 0.1; /// \brief Initial step size
      double minDelta = 1e-20; /// \brief Minimum step size
      double maxDelta = 1.0; /// \brief Maximum step size
      double convergenceGradientNorm = 1e-3; /// \brief Stopping criterion on gradient norm
      double convergenceDx = 0.0; /// \brief Stopping criterion on maximum state update coefficient
      double convergenceDObjective = 0.0; /// \brief Stopping criterion on change of objective/error
      int maxIterations = 20; /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria. -1
      std::size_t nThreads = 4; /// \brief The number of threads to use
      boost::shared_ptr<ScalarNonSquaredErrorTerm> regularizer = NULL; /// \brief Regularizer
      Method method = RPROP_PLUS; /// \brief the RProp method used

      void check() const;
    };

    std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options);

    struct RpropReturnValue {
      enum ConvergenceCriterion { IN_PROGRESS = 0, FAILURE, GRADIENT_NORM, DX, DOBJECTIVE };
      RpropReturnValue() { }
      void reset();
      bool success() const;
      bool failure() const;
      ConvergenceCriterion convergence = IN_PROGRESS;
      std::size_t nIterations = 0;
      std::size_t nGradEvaluations = 0;
      std::size_t nObjectiveEvaluations = 0;
      double gradientNorm = std::numeric_limits<double>::signaling_NaN();
      double maxDx = std::numeric_limits<double>::signaling_NaN();
      double error = std::numeric_limits<double>::max();
      double derror = std::numeric_limits<double>::signaling_NaN(); /// \brief last change of the error
    };
    std::ostream& operator<<(std::ostream& out, const RpropReturnValue::ConvergenceCriterion& convergence);

    /**
     * \class OptimizerRprop
     *
     * RPROP implementation for the ASLAM framework.
     */
    class OptimizerRprop : private ProblemManager {
    public:

      using ProblemManager::setProblem;
      using ProblemManager::checkProblemSetup;

      typedef boost::shared_ptr<OptimizerRprop> Ptr;
      typedef boost::shared_ptr<const OptimizerRprop> ConstPtr;
      typedef OptimizerRpropOptions Options;

      /// \brief Constructor with default options
      OptimizerRprop();
      /// \brief Constructor with custom options
      OptimizerRprop(const Options& options);
      /// \brief Constructor from property tree
      OptimizerRprop(const sm::PropertyTree& config);
      /// \brief Destructor
      ~OptimizerRprop();

      /// \brief Run the optimization
      const RpropReturnValue& optimize();

      /// \brief Return the status
      const RpropReturnValue& getStatus() const { return _returnValue; }

      /// \brief Get the optimizer options.
      Options& options() { return _options; }

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
      /// \brief branchless signum method
      static inline int sign(const double& val) {
        return (0.0 < val) - (val < 0.0);
      }

    private:

      /// \brief The dense update vector.
      ColumnVectorType _dx;

      /// \brief current step-length to be performed into the negative direction of the gradient
      ColumnVectorType _delta;

      /// \brief gradient in the previous iteration
      RowVectorType _prev_gradient;

      /// \brief error in the previous iteration (only used for IRPROP_PLUS version)
      double _prev_error = std::numeric_limits<double>::max();

      /// \brief the current set of options
      Options _options;

      /// \brief Struct returned by optimize method
      RpropReturnValue _returnValue;

    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
