#ifndef ASLAM_BACKEND_MATRIX_BASIC_HPP
#define ASLAM_BACKEND_MATRIX_BASIC_HPP

#include <Eigen/Core>
#include <aslam/backend/DesignVariable.hpp>
#include "MatrixExpression.hpp"
#include "MatrixExpressionNode.hpp"

namespace aslam {
namespace backend {

class MatrixBasic : public MatrixExpressionNode, public DesignVariable {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MatrixBasic(const Eigen::Matrix3d & A);
  MatrixBasic(const Eigen::Matrix3d & A, bool add_dv);
  MatrixBasic(const Eigen::Matrix3d & A, const Eigen::Matrix3i & UpdatePattern);

  ~MatrixBasic() override;

  /// \brief Revert the last state update.
  void revertUpdateImplementation() override;

  /// \brief Update the design variable.
  void updateImplementation(const double * dp, int size) override;

  /// \brief the size of an update step
  int minimalDimensionsImplementation() const override;

  MatrixExpression toExpression();
 private:
  Eigen::Matrix3d evaluateImplementation() const override;
  void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
  void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

  /// Returns the content of the design variable
  void getParametersImplementation(Eigen::MatrixXd& value) const override;

  /// Sets the content of the design variable
  void setParametersImplementation(const Eigen::MatrixXd& value) override;

  Eigen::Matrix3d _A;
  Eigen::Matrix3d _p_A;
  Eigen::Matrix<double, 9, Eigen::Dynamic> _B;  //9xN (N<=9) update matrix derived from updatePattern
};

}  // namespace backend
}  // namespace aslam

#endif /* ASLAM_BACKEND_MATRIX_BASIC_HPP */
