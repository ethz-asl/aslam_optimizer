#ifndef ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP
#include <aslam/backend/JacobianContainer.hpp>

namespace aslam {
  namespace backend {
    
    template<int D>
    class VectorExpressionNode
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef Eigen::Matrix<double,D,1> vector_t;

      VectorExpressionNode();
      virtual ~VectorExpressionNode();
      
      vector_t evaluate() const;
      vector_t toVector() const;
      
      void evaluateJacobians(JacobianContainer & outJacobians) const;
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      void getDesignVariables(DesignVariable::set_t & designVariables) const;

    private:
      virtual vector_t evaluateImplementation() const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const = 0;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;

    };

  } // namespace backend
} // namespace aslam


#include "implementation/VectorExpressionNode.hpp"

#endif /* ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP */
