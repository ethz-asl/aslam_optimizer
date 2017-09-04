#ifndef ASLAM_TRANSFORMATION_EXPRESSION_NODE
#define ASLAM_TRANSFORMATION_EXPRESSION_NODE

#include <Eigen/Core>
#include <aslam/backend/JacobianContainer.hpp>
#include <boost/shared_ptr.hpp>
#include <set>

namespace aslam {
  namespace backend {
    class EuclideanExpression;
    class RotationExpression;
    class ExpressionNodeVisitor;

    class TransformationExpressionNode {
    public:
      TransformationExpressionNode();
      virtual ~TransformationExpressionNode();

      /// \brief Evaluate the transformation matrix.
      Eigen::Matrix4d evaluate() { return toTransformationMatrixImplementation(); }
      Eigen::Matrix4d toTransformationMatrix() { return toTransformationMatrixImplementation(); }

      /// \brief Evaluate the Jacobians
      void evaluateJacobians(JacobianContainer & outJacobians) const;
      template <typename DERIVED>
      EIGEN_ALWAYS_INLINE void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixBase<DERIVED> & applyChainRule) const {
        evaluateJacobians(outJacobians.apply(applyChainRule));
      }
      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      virtual void accept(ExpressionNodeVisitor& visitor); //TODO make pure and complete nodes
    protected:
      // These functions must be implemented by child classes.
      virtual Eigen::Matrix4d toTransformationMatrixImplementation() = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;

      virtual RotationExpression toRotationExpression(const boost::shared_ptr<TransformationExpressionNode> & thisShared) const;
      virtual EuclideanExpression toEuclideanExpression(const boost::shared_ptr<TransformationExpressionNode> & thisShared) const;

      friend class TransformationExpression;
    };

    /**
     * \class TransformationExpressionNodeMultiply
     *
     * \brief A class representing the multiplication of two transformation matrices.
     * 
     */
    class TransformationExpressionNodeMultiply : public TransformationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      TransformationExpressionNodeMultiply(boost::shared_ptr<TransformationExpressionNode> lhs, 
                                           boost::shared_ptr<TransformationExpressionNode> rhs);

      ~TransformationExpressionNodeMultiply() override;

      void accept(ExpressionNodeVisitor& visitor) override;
    private:
      Eigen::Matrix4d toTransformationMatrixImplementation() override;
      void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
      void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

      boost::shared_ptr<TransformationExpressionNode> _lhs;
      Eigen::Matrix4d _T_lhs;
      boost::shared_ptr<TransformationExpressionNode> _rhs;
      Eigen::Matrix4d _T_rhs;
    };


    /**
     * \class TransformationExpressionNodeInverse
     * 
     * \brief A class representing the inverse of a transformation matrix.
     *
     */
    class TransformationExpressionNodeInverse : public TransformationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      TransformationExpressionNodeInverse(boost::shared_ptr<TransformationExpressionNode> dvTransformation);

      ~TransformationExpressionNodeInverse() override;

      void accept(ExpressionNodeVisitor& visitor) override;
    private:
      Eigen::Matrix4d toTransformationMatrixImplementation() override;
      void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
      void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

      boost::shared_ptr<TransformationExpressionNode> _dvTransformation;
      Eigen::Matrix4d _T;
    };


    /**
     * \class TransformationExpressionNodeMultiply
     *
     * \brief A class representing the multiplication of two transformation matrices.
     * 
     */
    class TransformationExpressionNodeConstant : public TransformationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      TransformationExpressionNodeConstant(const Eigen::Matrix4d & T);
      ~TransformationExpressionNodeConstant() override;

      void accept(ExpressionNodeVisitor& visitor) override;
    private:
      Eigen::Matrix4d toTransformationMatrixImplementation() override;
      void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
      void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;


      Eigen::Matrix4d _T;
    };

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_TRANSFORMATION_EXPRESSION_NODE */
