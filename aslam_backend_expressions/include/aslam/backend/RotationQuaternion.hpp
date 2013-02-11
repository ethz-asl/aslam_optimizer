#ifndef ASLAM_BACKEND_DV_QUAT_HPP
#define ASLAM_BACKEND_DV_QUAT_HPP


#include <Eigen/Core>
#include <aslam/backend/DesignVariable.hpp>
#include "RotationExpression.hpp"
#include "RotationExpressionNode.hpp"

namespace aslam {
  namespace backend {
    
    class RotationQuaternion : public RotationExpressionNode, public DesignVariable
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      RotationQuaternion(const Eigen::Vector4d & q);

      /// Constructs a rotation quaternion expression from a rotation matrix
      RotationQuaternion(const Eigen::Matrix3d& C);

      virtual ~RotationQuaternion();

      /// \brief Revert the last state update.
      virtual void revertUpdateImplementation();

      /// \brief Update the design variable.
      virtual void updateImplementation(const double * dp, int size);

      /// \brief the size of an update step
      virtual int minimalDimensionsImplementation() const;

      RotationExpression toExpression();
    private:
      virtual Eigen::Matrix3d toRotationMatrixImplementation();
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;
      
      Eigen::Vector4d _q;
      Eigen::Vector4d _p_q;
      Eigen::Matrix3d _C;
      
    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_DV_QUAT_HPP */
