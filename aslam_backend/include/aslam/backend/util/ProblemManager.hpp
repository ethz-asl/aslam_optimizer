/*
 * ProblemManager.hpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_PROBLEMMANAGER_HPP_
#define INCLUDE_ASLAM_BACKEND_PROBLEMMANAGER_HPP_

#include <vector>

#include <boost/shared_ptr.hpp>

#include "CommonDefinitions.hpp"

#include "../../Exceptions.hpp"

namespace aslam {
namespace backend {

// Forward declarations
class OptimizationProblemBase;
class ErrorTerm;
class ScalarNonSquaredErrorTerm;
class DesignVariable;

/**
 * \class ProblemManager
 * Utility class to collect functionality for dealing with a problem.
 */
class ProblemManager {
 public:
  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

  /// \brief Constructor with default options
  ProblemManager();
  /// \brief Destructor
  virtual ~ProblemManager();

  /// \brief Set up to work on the optimization problem.
  void setProblem(boost::shared_ptr<OptimizationProblemBase> problem);

  /// \brief initialize the class
  virtual void initialize();

  /// \brief Is everything initialized?
  bool isInitialized() const { return _isInitialized; }

  /// \brief Mutable getter for the optimization problem
  boost::shared_ptr<OptimizationProblemBase> getProblem() { return _problem; }

  /// \brief Const getter for the optimization problem
  boost::shared_ptr<const OptimizationProblemBase> getProblem() const { return _problem; }

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
  void checkProblemSetup() const;

  /// \brief Evaluate the value of the objective function
  double evaluateError() const;

  /// \brief Signal that the problem changed.
  void signalProblemChanged() { setInitialized(false); }

  /// \brief Apply the update vector to the design variables
  void applyStateUpdate(const ColumnVectorType& dx);

  /// \brief Undo the last state update to the design variables
  void revertLastStateUpdate();

  /// \brief compute the current gradient of the objective function
  void computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator);

 protected:
  /// \brief Set the initialized status
  void setInitialized(bool isInitialized) { _isInitialized = isInitialized; }

 private:
  /// \brief Evaluate the gradient of the objective function
  void evaluateGradients(size_t threadId, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& grad);

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

#endif /* INCLUDE_ASLAM_BACKEND_PROBLEMMANAGER_HPP_ */
