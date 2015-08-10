#ifndef ASLAM_BACKEND_OPTIMIZER_RPROP_HPP
#define ASLAM_BACKEND_OPTIMIZER_RPROP_HPP

#include <aslam/backend/ScalarOptimizerBase.hpp>

namespace sm {
  class PropertyTree;
}

namespace aslam {
  namespace backend {

    struct OptimizerRpropOptions {
      OptimizerRpropOptions();

      double etaMinus; /// \brief Decrease factor for step size if gradient direction changes
      double etaPlus; /// \brief Increase factor for step size if gradient direction is same
      double initialDelta; /// \brief Initial step size
      double minDelta; /// \brief Minimum step size
      double maxDelta; /// \brief Maximum step size
      double convergenceGradientNorm; /// \brief Stopping criterion on gradient norm
      int maxIterations; /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria.
      bool verbose; /// \brief should we print out some information each iteration?
      int nThreads; /// \brief The number of threads to use

    };

    std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options);


    /**
     * \class OptimizerRprop
     *
     * RPROP implementation for the ASLAM framework.
     */
    class OptimizerRprop : public ScalarOptimizerBase {
    public:
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

    private:

      /// \brief initialize the optimizer to run on an optimization problem.
      virtual void initializeImplementation();

    private:

      /// \brief The dense update vector.
      ColumnVectorType _dx;

      /// \brief current step-length to be performed into the negative direction of the gradient
      ColumnVectorType _delta;

      /// \brief gradient in the previous iteration
      RowVectorType _prev_gradient;

      /// \brief Current norm of the gradient
      double _curr_gradient_norm;

      /// \brief the current set of options
      OptimizerRpropOptions _options;

      /// \brief How many iterations the solver has run
      std::size_t _nIterations;

    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
