#ifndef ASLAM_OPTIMIZATION_PROBLEM_HPP
#define ASLAM_OPTIMIZATION_PROBLEM_HPP

#include <aslam/Exceptions.hpp>
#include <set>

namespace aslam {
  namespace backend {
    class DesignVariable;
    class ErrorTerm;

    class OptimizationProblemBase {
    public:
      SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

      OptimizationProblemBase();
      virtual ~OptimizationProblemBase();

      /// \brief The number of design variables stored in this optimization problem
      size_t numDesignVariables() const;

      /// \brief Get design variable i
      DesignVariable* designVariable(size_t i);

      /// \brief Get design variable i
      const DesignVariable* designVariable(size_t i) const;

      /// \brief the number of error terms stored in this optimization problem
      size_t numErrorTerms() const;

      /// \brief get error term i.
      ErrorTerm* errorTerm(size_t i);

      /// \brief get error term i.
      const ErrorTerm* errorTerm(size_t i) const;

      /// \brief Get the error terms associated with a specific design variable
      void getErrors(const DesignVariable* dv, std::set<ErrorTerm*>& outErrorSet);
    protected:

      /// \brief the number of design variables used in this optimization problem
      virtual size_t numDesignVariablesImplementation() const = 0;
      /// \brief get design variable k
      virtual DesignVariable* designVariableImplementation(size_t k) = 0;
      /// \brief get design variable k
      virtual const DesignVariable* designVariableImplementation(size_t k) const = 0;

      /// \brief the number of error terms in this optimization problem
      virtual size_t numErrorTermsImplementation() const = 0;
      /// \brief get error term n
      virtual ErrorTerm* errorTermImplementation(size_t n) = 0;
      /// \brief get error term n
      virtual const ErrorTerm* errorTermImplementation(size_t n) const = 0;

      /// \brief get all of the error terms associated with design variable dv.
      virtual void getErrorsImplementation(const DesignVariable* dv, std::set<ErrorTerm*>& outErrorSet) = 0;

    };


  } // namespace backend
} // namespace aslam


#endif /* ASLAM_OPTIMIZATION_PROBLEM_HPP */
