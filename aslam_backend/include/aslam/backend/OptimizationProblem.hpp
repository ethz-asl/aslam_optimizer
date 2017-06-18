#ifndef ASLAM_BACKEND_OPTIMIZATION_PROBLEM_SIMPLE
#define ASLAM_BACKEND_OPTIMIZATION_PROBLEM_SIMPLE

#include "OptimizationProblemBase.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>

#include <unordered_map>

namespace aslam {
  namespace backend {

    /**
     * \class OptimizationProblem
     *
     * \brief a simple implementation of the optimization problem that
     *        only stores containers of design variables and error terms.
     *        This container owns the design variables and error terms and
     *        it will call delete on them when it goes out of scope.
     */
    class OptimizationProblem : public OptimizationProblemBase {
    public:
      OptimizationProblem();
      ~OptimizationProblem() override;

      /// \brief Add a design variable to the problem. If the second
      /// argument is true, the design variable will be deleted
      /// when the problem is cleared or goes out of scope.
      void addDesignVariable(DesignVariable* dv, bool problemOwnsVariable);

      /// \brief Add a design variable to the problem.
      void addDesignVariable(boost::shared_ptr<DesignVariable> dv);

      /// \brief Remove the design variable. This may cause error terms to also be removed.
      void removeDesignVariable(const DesignVariable* dv);

      /// \brief Add an error term to the problem. If the second
      /// argument is true, the error term will be deleted when the
      /// problem is cleared or goes out of scope.
      void addErrorTerm(ErrorTerm* dv, bool problemOwnsVariable);

      /// \brief Add a scalar non-squared error term to the problem. If the second
      /// argument is true, the error term will be deleted when the
      /// problem is cleared or goes out of scope.
      void addErrorTerm(ScalarNonSquaredErrorTerm* dv, bool problemOwnsVariable);

      /// \brief Add an error term to the problem
      void addErrorTerm(const boost::shared_ptr<ErrorTerm> & et) override;

      /// \brief Add a scalar non-squared error term to the problem
      virtual void addErrorTerm(const boost::shared_ptr<ScalarNonSquaredErrorTerm> & et);

      /// \brief Remove the error term
      void removeErrorTerm(const ErrorTerm* dv);

      /// \brief clear the design variables and error terms.
      void clear();

      /// \brief used for debugging...is the design variable in the problem.
      bool isDesignVariableInProblem(const DesignVariable* dv);

      size_t countActiveDesignVariables();

    protected:
      size_t numDesignVariablesImplementation() const override;
      DesignVariable* designVariableImplementation(size_t i) override;
      const DesignVariable* designVariableImplementation(size_t i) const override;

      size_t numErrorTermsImplementation() const override;
      size_t numNonSquaredErrorTermsImplementation() const override;
      ErrorTerm* errorTermImplementation(size_t i) override;
      ScalarNonSquaredErrorTerm* nonSquaredErrorTermImplementation(size_t i) override;
      const ErrorTerm* errorTermImplementation(size_t i) const override;
      const ScalarNonSquaredErrorTerm* nonSquaredErrorTermImplementation(size_t i) const override;

      void getErrorsImplementation(const DesignVariable* dv, std::set<ErrorTerm*>& outErrorSet) override;
      void getNonSquaredErrorsImplementation(const DesignVariable* dv, std::set<ScalarNonSquaredErrorTerm*>& outErrorSet) override;

      // \todo Replace these std::vectors by something better. The underlying algorithms that this object
      //       supports suck with these containers. Blerg. See "removeDesignVariable()" for an example of
      //       just how bad this is.
      std::vector< boost::shared_ptr<DesignVariable> > _designVariables;
      std::vector< boost::shared_ptr<ErrorTerm> > _errorTerms;
      std::vector< boost::shared_ptr<ScalarNonSquaredErrorTerm> > _sNSErrorTerms;
      typedef std::unordered_multimap< DesignVariable*, ErrorTerm*> error_map_t;
      typedef std::unordered_multimap< DesignVariable*, ScalarNonSquaredErrorTerm*> error_map_sns_t;
      error_map_t _errorTermMap;
      error_map_sns_t _errorTermMapSns;
    };


  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_OPTIMIZATION_PROBLEM_SIMPLE */
