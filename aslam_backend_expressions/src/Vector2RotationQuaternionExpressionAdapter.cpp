/*
 * Vector2Vector2RotationQuaternionExpressionAdapterExpressionAdapter.cpp
 *
 *  Created on: Jul 11, 2013
 *      Author: hannes
 */


#include <aslam/backend/Vector2RotationQuaternionExpressionAdapter.hpp>

namespace aslam {
namespace backend {

  Vector2RotationQuaternionExpressionAdapter::Vector2RotationQuaternionExpressionAdapter(const VectorExpression<4> & vectorExpression) :
      _root(vectorExpression.root())
  {
  }

  Vector2RotationQuaternionExpressionAdapter::~Vector2RotationQuaternionExpressionAdapter(){}

  Eigen::Matrix3d Vector2RotationQuaternionExpressionAdapter::toRotationMatrixImplementation()
  {
    return sm::kinematics::quat2r(_root->toVector());
  }

  void Vector2RotationQuaternionExpressionAdapter::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
  {
    _root->evaluateJacobians(outJacobians);
  }

  void Vector2RotationQuaternionExpressionAdapter::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
  {
    _root->evaluateJacobians(outJacobians, applyChainRule);
  }

  void Vector2RotationQuaternionExpressionAdapter::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
  {
    _root->getDesignVariables(designVariables);
  }
} // namespace backend

} /* namespace aslam */
