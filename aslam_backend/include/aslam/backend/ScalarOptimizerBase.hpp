/*
 * ScalarOptimizerBase.hpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_SCALAROPTIMIZERBASE_HPP_
#define INCLUDE_ASLAM_BACKEND_SCALAROPTIMIZERBASE_HPP_

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <Eigen/Core>

#include <sm/timing/Timer.hpp>

#include "../Exceptions.hpp"

namespace aslam {
  namespace backend {

    class OptimizationProblemBase;
    class ErrorTerm;
    class ScalarNonSquaredErrorTerm;
    class DesignVariable;

    /**
     * \class ScalarOptimizerBase
     * Interface definition for optimizers working on scalar objective functions
     */
    class ScalarOptimizerBase {
    public:
#ifdef aslam_backend_ENABLE_TIMING
      typedef sm::timing::Timer Timer;
#else
      typedef sm::timing::DummyTimer Timer;
#endif
      typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
      typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorType;

      SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

      /// \brief Constructor with default options
      ScalarOptimizerBase();
      /// \brief Destructor
      virtual ~ScalarOptimizerBase();

      /// \brief Set up to work on the optimization problem.
      void setProblem(boost::shared_ptr<OptimizationProblemBase> problem);

      /// \brief initialize the optimizer to run on an optimization problem.
      void initialize();

      /// \brief Is everything initialized?
      bool isInitialized() const { return _isInitialized; }

      /// \brief Mutable getter for the optimization problem
      boost::shared_ptr<OptimizationProblemBase> getProblem() { return _problem; }

      /// \brief Get dense design variable i.
      DesignVariable* designVariable(size_t i);

      /// \brief how many dense design variables are involved in the problem
      size_t numDesignVariables() const { return _designVariables.size(); };

      /// \brief how many scalar parameters (design variables with their minimal dimension)
      ///        are involved in the problem
      size_t numOptParameters() const { return _numOptParameters; }

      /// \brief how many error terms are involved in the problem
      size_t numErrorTerms() const { return _numErrorTerms; }

      /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
      ///        hooked up to design variables and running finite differences on error terms where this is possible.
      void checkProblemSetup();

    protected:

      void setInitialized(bool isInitialized) { _isInitialized = isInitialized; }

      /// \brief Additional initialization functionality by child classes
      virtual void initializeImplementation() { }

      /// \brief compute the current gradient of the objective function
      void computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator);

      double evaluateError() const;

      /// \brief Apply the update vector to the design variables
      void applyStateUpdate(const ColumnVectorType& dx);

      /// \brief Undo the last state update to the design variables
      void revertLastStateUpdate();

    private:
      void evaluateGradients(size_t threadId, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& grad);
      void setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job,
                            size_t nThreads,
                            std::vector<RowVectorType>& out,
                            bool useMEstimator);

    private:

      /// \brief The current optimization problem.
      boost::shared_ptr<OptimizationProblemBase> _problem;

      /// \brief all design variables...first the non-marginalized ones (the dense ones), then the marginalized ones.
      std::vector<DesignVariable*> _designVariables;

      /// \brief all of the error terms involved in this problem
      std::vector<ErrorTerm*> _errorTermsS;
      std::vector<ScalarNonSquaredErrorTerm*> _errorTermsNS;

      /// \brief the total number of parameters of this problem, given by number of design variables and their dimensionality
      std::size_t _numOptParameters;

      /// \brief the total number of error terms as the sum of squared and non-squared error terms
      std::size_t _numErrorTerms;

      /// \brief Whether the optimizer is correctly initialized
      bool _isInitialized;

    };

  } // namespace backend
} // namespace aslam

#endif /* INCLUDE_ASLAM_BACKEND_SCALAROPTIMIZERBASE_HPP_ */
