#ifndef ASLAM_BACKEND_VECTOR_EXPRESSION_HPP
#define ASLAM_BACKEND_VECTOR_EXPRESSION_HPP

#include <aslam/backend/JacobianContainer.hpp>
#include <boost/shared_ptr.hpp>
#include <sm/boost/null_deleter.hpp>
#include "VectorExpressionNode.hpp"
#include <aslam/backend/ScalarExpression.hpp>
#include <aslam/backend/ScalarExpressionNode.hpp>

namespace aslam {
  namespace backend {
    

    template<int D>
    class VectorExpression
    {
    public:
      typedef Eigen::Matrix<double,D,1> vector_t;
      typedef Eigen::Matrix<double,D,1> value_t;

      VectorExpression() = default;
      VectorExpression(boost::shared_ptr< VectorExpressionNode<D> > root);
      VectorExpression(VectorExpressionNode<D> * root);
      VectorExpression(const vector_t & v);
      
      vector_t evaluate() const;
      vector_t toValue() const { return evaluate(); }
      
      void evaluateJacobians(JacobianContainer & outJacobians) const;
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      ScalarExpression toScalarExpression() const;
      template <int ColumnIndex>
      ScalarExpression toScalarExpression() const;

      boost::shared_ptr< VectorExpressionNode<D> > root() const { return _root; }

      bool isEmpty() const { return !_root; }
    protected:
      boost::shared_ptr< VectorExpressionNode<D> > _root;
    };

  } // namespace backend
} // namespace aslam

#include "implementation/VectorExpression.hpp"

#endif /* ASLAM_BACKEND_VECTOR_EXPRESSION_HPP */
