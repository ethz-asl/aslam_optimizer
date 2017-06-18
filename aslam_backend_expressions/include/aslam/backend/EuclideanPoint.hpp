#ifndef ASLAM_BACKEND_EUCLIDEAN_POINT_HPP
#define ASLAM_BACKEND_EUCLIDEAN_POINT_HPP

#include "EuclideanExpressionNode.hpp"
#include "EuclideanExpression.hpp"
#include <aslam/backend/DesignVariable.hpp>


namespace aslam {
  namespace backend {
  class HomogeneousExpression;
  
    class EuclideanPoint : public EuclideanExpressionNode, public DesignVariable
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      EuclideanPoint(const Eigen::Vector3d & p);
      ~EuclideanPoint() override;

      /// \brief Revert the last state update.
      void revertUpdateImplementation() override;

      /// \brief Update the design variable.
      void updateImplementation(const double * dp, int size) override;

      /// \brief the size of an update step
      int minimalDimensionsImplementation() const override;

      EuclideanExpression toExpression();
      HomogeneousExpression toHomogeneousExpression();

      void set(const Eigen::Vector3d & p){ _p = p; _p_p = _p; }

      const Eigen::Vector3d & getValue() const { return _p; }
      const Eigen::Vector3d & toEuclidean() const { return getValue() ; }
    private:
      Eigen::Vector3d evaluateImplementation() const override;

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

      /// \brief The current value of the design variable.
      Eigen::Vector3d _p;

      /// \brief The previous version of the design variable.
      Eigen::Vector3d _p_p;
    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_EUCLIDEAN_POINT_HPP */
