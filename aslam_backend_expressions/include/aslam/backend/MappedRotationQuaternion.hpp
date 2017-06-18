#ifndef ASLAM_BACKEND_MAPPED_DV_QUAT_HPP
#define ASLAM_BACKEND_MAPPED_DV_QUAT_HPP


#include <Eigen/Core>
#include <aslam/backend/DesignVariable.hpp>
#include "RotationExpression.hpp"
#include "RotationExpressionNode.hpp"

namespace aslam {
  namespace backend {
    
    class MappedRotationQuaternion : public RotationExpressionNode, public DesignVariable
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      MappedRotationQuaternion(double * q);

      ~MappedRotationQuaternion() override;

      /// \brief Revert the last state update.
      void revertUpdateImplementation() override;

      /// \brief Update the design variable.
      void updateImplementation(const double * dp, int size) override;

      /// \brief the size of an update step
      int minimalDimensionsImplementation() const override;

      RotationExpression toExpression();

      void set( const Eigen::Vector4d & q){ _q = q; _p_q = q; }
    private:
      Eigen::Matrix3d toRotationMatrixImplementation() const override;
      void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
      void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

      /// Returns the content of the design variable
      void getParametersImplementation(Eigen::MatrixXd& value) const override;

      /// Sets the content of the design variable
      void setParametersImplementation(const Eigen::MatrixXd& value) override;

      /// Computes the minimal distance in tangent space between the current value of the DV and xHat
      void minimalDifferenceImplementation(const Eigen::MatrixXd& xHat, Eigen::VectorXd& outDifference) const override;

      /// Computes the minimal distance in tangent space between the current value of the DV and xHat and the jacobian
      void minimalDifferenceAndJacobianImplementation(const Eigen::MatrixXd& xHat, Eigen::VectorXd& outDifference, Eigen::MatrixXd& outJacobian) const override;

      Eigen::Map<Eigen::Vector4d> _q;
      Eigen::Vector4d _p_q;
      Eigen::Matrix3d _C;
    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_DV_QUAT_HPP */
