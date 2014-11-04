#ifndef ASLAM_BACKEND_DV_ROTATION_HPP
#define ASLAM_BACKEND_DV_ROTATION_HPP


#include <aslam/backend/JacobianContainer.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>
#include <boost/shared_ptr.hpp>
#include <set>
#include <aslam/backend/TransformationExpressionNode.hpp>

namespace aslam {
  namespace backend {
    
    /**
     * \class RotationExpressionNode
     * \brief The superclass of all classes representing rotations.
     */
    class RotationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      RotationExpressionNode();
      virtual ~RotationExpressionNode();

      /// \brief Evaluate the rotation matrix.
      Eigen::Matrix3d toRotationMatrix() const;
      
      /// \brief Evaluate the Jacobians
      void evaluateJacobians(JacobianContainer & outJacobians) const;
    
      /// \brief Evaluate the Jacobians and apply the chain rule.
      /** The chain rule matrix is assumed to be calculated in the left exponential chart centered at the current value (Phi(w)=Phi_R(w):= exp(w) R)),
       * where exp(w) applied to 3x1 vectors is the matrix exponential of the skew symmetric matrix w^x,
       * such that, assuming that this RotataionExpression, R, is the argument of a function g:
       * J g\circ R = J (g \circ Phi \circ Phi^{-1} \circ R) = J( g \circ Phi) J( Phi^{-1} \circ R)
       * applyChainRule := J(g \circ Phi);
       *
       * This implies e.g. for the product A * B of two rotation expression that:
       * evaluateJacobiansImplementation assumes that
       * applyChainRule = J ( g * Phi_{A * B} } and does return
       * J_v,w ( Phi_{A*B}^{-1} \circ (Phi_A(v) * Phi_B(w))) = J_v,w ( log( (exp(v) A * exp(w)B B^{-1}A^{-1}) ))
       * such that
       * J_v = id and
       * J_w = J_w (log (A * exp(w) * A^{-1}) = log (exp(A * w)) = A
       */
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      void getDesignVariables(DesignVariable::set_t & designVariables) const;
    protected:        
      // These functions must be implemented by child classes.
      virtual Eigen::Matrix3d toRotationMatrixImplementation() const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const = 0;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;
    };

    /**
     * \class ConstantRotationExpression
     *
     * \brief A class representing a constant rotation matrix.
     *
     */
    class ConstantRotationExpressionNode : public RotationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      ConstantRotationExpressionNode(const Eigen::Matrix3d & C);
      virtual ~ConstantRotationExpressionNode();

    private:
      virtual Eigen::Matrix3d toRotationMatrixImplementation() const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      const Eigen::Matrix3d _C;
    };

    /**
     * \class RotationExpressionNodeMultiply
     *
     * \brief A class representing the multiplication of two rotation matrices.
     * 
     */
    class RotationExpressionNodeMultiply : public RotationExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      RotationExpressionNodeMultiply(boost::shared_ptr<RotationExpressionNode> lhs, 
				     boost::shared_ptr<RotationExpressionNode> rhs);
      virtual ~RotationExpressionNodeMultiply();

    private:
      virtual Eigen::Matrix3d toRotationMatrixImplementation() const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      boost::shared_ptr<RotationExpressionNode> _lhs;
      mutable Eigen::Matrix3d _C_lhs;
      boost::shared_ptr<RotationExpressionNode> _rhs;
      mutable Eigen::Matrix3d _C_rhs;
    };


    /**
     * \class RotationExpressionNodeInverse
     * 
     * \brief A class representing the inverse of a rotation matrix.
     *
     */
    class RotationExpressionNodeInverse : public RotationExpressionNode
    {
    public:
      RotationExpressionNodeInverse(boost::shared_ptr<RotationExpressionNode> dvRotation);

      virtual ~RotationExpressionNodeInverse();

    private:
      virtual Eigen::Matrix3d toRotationMatrixImplementation() const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      boost::shared_ptr<RotationExpressionNode> _dvRotation;
      mutable Eigen::Matrix3d _C;
    };

    class RotationExpressionNodeTransformation : public RotationExpressionNode
    {
    public:
      RotationExpressionNodeTransformation(boost::shared_ptr<TransformationExpressionNode> dvRotation);

      virtual ~RotationExpressionNodeTransformation();

    private:
      virtual Eigen::Matrix3d toRotationMatrixImplementation() const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      boost::shared_ptr<TransformationExpressionNode> _transformation;
    };

  

  } // namespace backend
} // namespace aslam



#endif /* ASLAM_BACKEND_DV_ROTATION_HPP */
