#ifndef ASLAM_BACKEND_SCALAR_POINT_HPP
#define ASLAM_BACKEND_SCALAR_POINT_HPP

#include "ScalarExpressionNode.hpp"
#include "ScalarExpression.hpp"
#include <aslam/backend/DesignVariable.hpp>


namespace aslam {
  namespace backend {
    
    class Scalar : public ScalarExpressionNode, public DesignVariable
    {
    public:

    	enum {
    		DesignVariableDimension = 1
    	};

      Scalar(const double & p);
      virtual ~Scalar();

      /// \brief Revert the last state update.
      virtual void revertUpdateImplementation();

      /// \brief Update the design variable.
      virtual void updateImplementation(const double * dp, int size);

      /// \brief the size of an update step
      virtual int minimalDimensionsImplementation() const;

      ScalarExpression toExpression();

      Eigen::MatrixXd getParameters();
        
    private:
      virtual double toScalarImplementation() const;

      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;

      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      /// \brief The current value of the design variable.
      double _p;

      /// \brief The previous version of the design variable.
      double _p_p;


    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_SCALAR_POINT_HPP */
