#ifndef ASLAM_BACKEND_HOMOGENEOUS_POINT_HPP
#define ASLAM_BACKEND_HOMOGENEOUS_POINT_HPP

#include "HomogeneousExpressionNode.hpp"
#include "HomogeneousExpression.hpp"

#include <aslam/backend/DesignVariable.hpp>

namespace aslam {
  namespace backend {
    
    class HomogeneousPoint : public HomogeneousExpressionNode, public DesignVariable
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      HomogeneousPoint(const Eigen::Vector4d & p);
      ~HomogeneousPoint() override;

      /// \brief Revert the last state update.
      void revertUpdateImplementation() override;

      /// \brief Update the design variable.
      void updateImplementation(const double * dp, int size) override;

      /// \brief the size of an update step
      int minimalDimensionsImplementation() const override;

      HomogeneousExpression toExpression();
    private:
      Eigen::Vector4d toHomogeneousImplementation() const override;

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

    private:
      /// \brief The current value of the design variable.
      Eigen::Vector4d _p;

      /// \brief The previous version of the design variable.
      Eigen::Vector4d _p_p;


    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_HOMOGENEOUS_POINT_HPP */
