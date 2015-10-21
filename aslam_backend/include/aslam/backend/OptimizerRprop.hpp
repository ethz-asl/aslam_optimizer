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
      int maxIterations = 20; /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria. -1
      std::size_t nThreads = 4; /// \brief The number of threads to use
      boost::shared_ptr<ScalarNonSquaredErrorTerm> regularizer = NULL; /// \brief Regularizer
      Method method = RPROP_PLUS; /// \brief the RProp method used

      void check() const;
    };

    std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options);


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

      /// \brief Constructor with default options
      OptimizerRprop();
      /// \brief Constructor with custom options
      OptimizerRprop(const OptimizerRpropOptions& options);
      /// \brief Constructor from property tree
      OptimizerRprop(const sm::PropertyTree& config);
      /// \brief Destructor
      ~OptimizerRprop();

      /// \brief Run the optimization
      void optimize();

      /// \brief Get the optimizer options.
      OptimizerRpropOptions& options() { return _options; }

      /// \brief Return the current gradient norm
      inline double getGradientNorm() { return _curr_gradient_norm; }

      /// \brief Get the number of iterations the solver has run.
      ///        If it has never been started, the value will be zero.
      inline std::size_t getNumberOfIterations() const { return _nIterations; }

      /// \brief Initialize the optimizer to run on an optimization problem.
      ///        optimize() will call initialize() upon the first call.
      virtual void initialize() override;

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

      /// \brief Current norm of the gradient
      double _curr_gradient_norm = std::numeric_limits<double>::signaling_NaN();

      /// \brief the current set of options
      OptimizerRpropOptions _options;

      /// \brief How many iterations the solver has run
      std::size_t _nIterations = 0;

    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
