#ifndef ASLAM_TRANSFORMATION_EXPRESSION_HPP
#define ASLAM_TRANSFORMATION_EXPRESSION_HPP

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <aslam/backend/JacobianContainer.hpp>
#include <sm/kinematics/Transformation.hpp>
#include "RotationExpression.hpp"
#include "HomogeneousExpression.hpp"
#include "TransformationExpression.hpp"
#include <set>

namespace aslam {
  namespace backend {
    class TransformationExpressionNode;
    class RotationExpression;
    class HomogeneousExpression;
    class EuclideanExpression;
    class ExpressionNodeVisitor;

    class TransformationExpression {
    public:
      TransformationExpression();
      TransformationExpression(const RotationExpression & rotation, const EuclideanExpression & translation);
      TransformationExpression(TransformationExpressionNode * root);
      TransformationExpression(boost::shared_ptr<TransformationExpressionNode> root);

      // create a constant expression
      TransformationExpression(const Eigen::Matrix4d & T);

      virtual ~TransformationExpression();

      Eigen::Matrix4d toTransformationMatrix() const;
      sm::kinematics::Transformation evaluate() const;
      sm::kinematics::Transformation toTransformation() const;

      RotationExpression toRotationExpression() const;
      EuclideanExpression toEuclideanExpression() const;

      void evaluateJacobians(JacobianContainer & outJacobians) const;
      template <typename DERIVED>
      EIGEN_ALWAYS_INLINE void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixBase<DERIVED> & applyChainRule) const {
        evaluateJacobians(outJacobians.apply(applyChainRule));
      }

      EuclideanExpression operator*(const EuclideanExpression & rhs) const;
      HomogeneousExpression operator*(const HomogeneousExpression & rhs) const;
      TransformationExpression operator*(const TransformationExpression & rhs) const;

      TransformationExpression inverse() const;

      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      boost::shared_ptr<TransformationExpressionNode> root(){ return _root; }

      virtual void accept(ExpressionNodeVisitor& visitor) const;
    private:
      boost::shared_ptr<TransformationExpressionNode> _root;
    };

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_TRANSFORMATION_EXPRESSION_HPP */
