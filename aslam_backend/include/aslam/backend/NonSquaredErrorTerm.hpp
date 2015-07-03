#ifndef ASLAM_NON_SQUARED_ERROR_TERM_HPP
#define ASLAM_NON_SQUARED_ERROR_TERM_HPP

#include <sparse_block_matrix/sparse_block_matrix.h>
#include <boost/shared_ptr.hpp>
#include "backend.hpp"
#include "JacobianContainer.hpp"
#include <aslam/Exceptions.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <sm/timing/Timer.hpp>
#include "MEstimatorPolicies.hpp"
#include <sm/eigen/matrix_sqrt.hpp>
#include <sm/timing/NsecTimeUtilities.hpp>

namespace aslam {
  namespace backend {
    class MEstimator;

    /**
     * \class NonSquaredErrorTerm
     *
     * \brief a class representing a single scalar error term in a general
     * optimization problem of the form \f$ \sum_{i=0}^N w_i \cdot e_i \f$.
     */
    class NonSquaredErrorTerm {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

      typedef boost::shared_ptr<aslam::backend::NonSquaredErrorTerm> Ptr;

      NonSquaredErrorTerm();
      virtual ~NonSquaredErrorTerm();

      /// \brief evaluate the error term and return the effective error.
      ///        This is equivalent to first call updateRawError() and then taking the result of getError();
      ///        After this is called, the _error is filled in with \f$ w \cdot e \f$
      double evaluateError() {
        updateRawError();
        return getError();
      }

      /// \brief update (compute and store) the raw error
      ///        After this is called, the _error is filled in with \f$ w \cdot e \f$
      double updateRawError();

      /// \brief Get the current, weighted error, i.e. with the M-estimator weight already applied.
      double getError();

      /// \brief evaluate the Jacobians.
      void evaluateJacobians(JacobianContainer & outJacobians);

      /// \brief evaluate the Jacobians using finite differences.
      void evaluateJacobiansFiniteDifference(JacobianContainer & outJacobians);
      
      inline virtual void getWeightedJacobians(JacobianContainer& outJc, bool useMEstimator);

      /// \brief Get the error (before weighting by the M-estimator policy)
      inline virtual double getWeightedError(bool useMEstimator) const;
      /// \brief Get the error (before weighting by the M-estimator policy)
      double getRawError() const;
      /// \brief Get the error (weighted by the M-estimator policy)
      double getWeightedError() const;
      /// \brief Get the error with or without M-estimator policy
      inline double getError(bool useMEstimator) const {
        return useMEstimator ? getWeightedError() : getRawError();
      }

      inline double getWeight() const;

      /// \brief returns the weight
      inline void setWeight(const double w);

      /// \brief get the current value of the error.
      /// This was put here to make the python interface easier to generate. It doesn't
      /// fit for quadratic integral terms so it may go away in future versions.
//      Eigen::VectorXd vsError() const;
//
//      virtual void getWeight(double& invR) const = 0;
//      virtual Eigen::MatrixXd vsInvR() const = 0;
//      virtual void vsSetInvR(const Eigen::MatrixXd& invR) = 0;

      /// \brief returns a pointer to the MEstimator used. Return Null if the
      /// MEstimator used is not a MEstimatorType
      template <typename MEstimatorType>
      boost::shared_ptr<MEstimatorType> getMEstimatorPolicy();

      /// \brief set the M-Estimator policy. This function takes a squared error
      ///        and returns a weight to apply to that error term.
      void setMEstimatorPolicy(const boost::shared_ptr<MEstimator> & mEstimator);

      /// \brief clear the m-estimator policy.
      void clearMEstimatorPolicy();

      /// \brief compute the M-estimator weight from a squared error.
      double getMEstimatorWeight(double error) const;

      double getCurrentMEstimatorWeight() const;

      /// \brief get the name of the M-Estimator.
      std::string getMEstimatorName();

      /// \brief build this error term's part of the Hessian matrix.
      ///
      /// the i/o variables outHessian and outRhs are the full Hessian and rhs in the Gauss-Newton
      /// problem. The correct blocks for each design varible are available from the design
      /// variable as dv.blockIndex()
//      void buildHessian(SparseBlockMatrix& outHessian, Eigen::VectorXd& outRhs, bool useMEstimator) ;

      /// \brief How many design variables is this error term connected to?
      size_t numDesignVariables() const;

      /// \brief Get design variable i.
      DesignVariable* designVariable(size_t i);

      /// \brief Get design variable i.
      const DesignVariable* designVariable(size_t i) const;

      /// \brief Fill the set with all design variables.
      void getDesignVariables(DesignVariable::set_t& dvs);

      /// \brief Get the design variables
      const std::vector<DesignVariable*> & designVariables() const;

      void setTime(const sm::timing::NsecTime& t);
      sm::timing::NsecTime getTime() { return _timestamp; }

    protected:

      /// \brief evaluate the error term and return the weighted squared error e^T invR e
      virtual double evaluateErrorImplementation() = 0;

      /// \brief evaluate the Jacobians
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) = 0;

      /// \brief set the error vector.
      inline void setError(const double e);

      /// \brief build this error term's part of the Hessian matrix.
      ///
      /// the i/o variables outHessian and outRhs are the full Hessian and rhs in the Gauss-Newton
      /// problem. The correct blocks for each design variable are available from the design
      /// variable as dv.blockIndex()
//      virtual void buildHessianImplementation(SparseBlockMatrix& outHessian, Eigen::VectorXd& outRhs, bool useMEstimator) = 0;

//      virtual double vsErrorImplementation() const = 0;

      /// \brief child classes should set the set of design variables using this function.
      void setDesignVariables(const std::vector<DesignVariable*> & designVariables);

      void setDesignVariables(DesignVariable* dv1);
      void setDesignVariables(DesignVariable* dv1, DesignVariable* dv2);
      void setDesignVariables(DesignVariable* dv1, DesignVariable* dv2, DesignVariable* dv3);
      void setDesignVariables(DesignVariable* dv1, DesignVariable* dv2, DesignVariable* dv3, DesignVariable* dv4);


      template<typename ITERATOR_T>
      void setDesignVariablesIterator(ITERATOR_T start, ITERATOR_T end);


      /// \brief the MEstimator policy for this error term
      boost::shared_ptr<MEstimator> _mEstimatorPolicy;

    private:
      /// \brief the error \f$ w \cdot e \f$
      double _error;

      /// \brief the weight in \f$ w \cdot e \f$
      double _w;

      /// \brief The list of design variables.
      std::vector<DesignVariable*> _designVariables;

//      size_t _rowBase;

      sm::timing::NsecTime _timestamp;
    };


  } // namespace backend
} // namespace aslam

#include "implementation/NonSquaredErrorTerm.hpp"

#endif /* ASLAM_NON_SQUARED_ERROR_TERM_HPP */
