#ifndef ASLAM_BACKEND_OPTIMIZER_RPROP_HPP
#define ASLAM_BACKEND_OPTIMIZER_RPROP_HPP


#include <boost/shared_ptr.hpp>
//#include <boost/function.hpp>
#include <sm/assert_macros.hpp>
#include <Eigen/Core>
#include "OptimizerRpropOptions.hpp"
#include "backend.hpp"
#include "OptimizationProblemBase.hpp"
#include <aslam/Exceptions.hpp>
#include <aslam/backend/NonSquaredErrorTerm.hpp>
#include <aslam/backend/Matrix.hpp>
#include <aslam/backend/LevenbergMarquardtTrustRegionPolicy.hpp>
#include <aslam/backend/DogLegTrustRegionPolicy.hpp>
#include <sm/timing/Timer.hpp>
#include <boost/thread.hpp>
#include <sparse_block_matrix/linear_solver.h>

namespace sm {
  class PropertyTree;
}

namespace aslam {
  namespace backend {
    class LinearSystemSolver;

    /**
     * \class OptimizerRprop
     *
     * RPROP implementation for the ASLAM framework.
     */
    class OptimizerRprop {
    public:
        //  typedef sm::timing::Timer Timer;
      /// Swapping this to the dummy timer will disable timing
      typedef sm::timing::DummyTimer Timer;
      typedef sparse_block_matrix::SparseBlockMatrix<Eigen::MatrixXd> SparseBlockMatrix;
      typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
      typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorType;

      SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

      OptimizerRprop(const OptimizerRpropOptions& options);
      OptimizerRprop(const sm::PropertyTree& config, boost::shared_ptr<LinearSystemSolver> linearSystemSolver, boost::shared_ptr<TrustRegionPolicy> trustRegionPolicy);
      ~OptimizerRprop();

      /// \brief Set up to work on the optimization problem.
      void setProblem(boost::shared_ptr<OptimizationProblemBase> problem);

      /// \brief initialize the optimizer to run on an optimization problem.
      void initialize();

      /// \brief Run the optimization
      void optimize();

      /// \brief Get the optimizer options.
      OptimizerRpropOptions& options();

      /// \brief return the reduced system dx
//      const Eigen::VectorXd& dx() const;

      /// The value of the objective function.
//      double J() const;

      /// \brief compute the current gradient of the objective function
      void computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator);

      /// \brief Return the current gradient norm
      double getGradientNorm() { return _curr_gradient_norm; }

      /// \brief Evaluate the error at the current state.
//      double evaluateError(bool useMEstimator);

      /// \brief Get dense design variable i.
      DesignVariable* designVariable(size_t i);

      /// \brief how many dense design variables are involved in the problem
      size_t numDesignVariables() const;

      /// \brief print the internal timing information.
      void printTiming() const;

      /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
      ///        hooked up to design variables and running finite differences on error terms where this is possible.
      void checkProblemSetup();

    private:
      void evaluateGradients(size_t threadId, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& grad);
      void setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job, size_t nThreads, std::vector<RowVectorType>& out, bool useMEstimator);

    private:

      /// \brief Revert the last state update.
      void revertLastStateUpdate();

      /// \brief Apply a state update.
      double applyStateUpdate(const ColumnVectorType& dx);

      /// \brief The dense update vector.
      ColumnVectorType _dx;

      /// \brief The current value of the gradient.
//      double _grad;

      /// \brief The previous value of the gradient.
//      double _p_grad;

      /// \brief current step-length to be performed into the negative direction of the gradient
      ColumnVectorType _delta;

      /// \brief gradient in the previous iteration
      RowVectorType _prev_gradient;

      /// \brief Current norm of the gradient
      double _curr_gradient_norm;

//      boost::shared_ptr<LinearSystemSolver> _solver;

//      boost::shared_ptr<TrustRegionPolicy> _trustRegionPolicy;

      /// \brief The current optimization problem.
      boost::shared_ptr<OptimizationProblemBase> _problem;

      /// \brief all design variables...first the non-marginalized ones (the dense ones), then the marginalized ones.
      std::vector<DesignVariable*> _designVariables;

      /// \brief all of the error terms involved in this problem
      std::vector<ErrorTerm*> _errorTermsS;
      std::vector<NonSquaredErrorTerm*> _errorTermsNS;

      /// \brief the current set of options
      OptimizerRpropOptions _options;

      /// \brief the total number of parameters of this problem, given by number of design variables and their dimensionality
      std::size_t _numOptParameters;

      /// \brief the total number of error terms as the sum of squared and non-squared error terms
      std::size_t _numErrorTerms;

      /// \brief Whether the optimizer is correctly initialized
      bool _isInitialized;

    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_HPP */
