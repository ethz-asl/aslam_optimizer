#ifndef ASLAM_BACKEND_SCALAR_POINT_HPP
#define ASLAM_BACKEND_SCALAR_POINT_HPP

#include "ScalarExpressionNode.hpp"
#include "ScalarExpression.hpp"
#include <aslam/backend/DesignVariable.hpp>

namespace aslam {
namespace backend {

class Scalar : public ScalarExpressionNode, public DesignVariable {
 public:

  enum {
    DesignVariableDimension = 1
  };

  Scalar(const double & p);
  ~Scalar() override;

  /// \brief Revert the last state update.
  void revertUpdateImplementation() override;

  /// \brief Update the design variable.
  void updateImplementation(const double * dp, int size) override;

  /// \brief the size of an update step
  int minimalDimensionsImplementation() const override;

  ScalarExpression toExpression();

  Eigen::MatrixXd getParameters();

  double getValue() const { return _p; }
  void setValue(double p) { _p = p; }
 private:
  double evaluateImplementation() const override;

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
  double _p;

  /// \brief The previous version of the design variable.
  double _p_p;

};

}  // namespace backend
}  // namespace aslam

#endif /* ASLAM_BACKEND_SCALAR_POINT_HPP */
