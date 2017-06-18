/*
 * Vector2RotationQuaternionExpressionAdapter.h
 *
 *  Created on: Jul 11, 2013
 *      Author: hannes
 */

#ifndef VECTOR2ROTATIONQUATERNIONEXPRESSIONADAPTER_H_
#define VECTOR2ROTATIONQUATERNIONEXPRESSIONADAPTER_H_

#include "RotationExpressionNode.hpp"
#include "RotationExpression.hpp"
#include "VectorExpression.hpp"


namespace aslam {
namespace backend {

class Vector2RotationQuaternionExpressionAdapter : public RotationExpressionNode {
 public:
  inline static RotationExpression adapt(const VectorExpression<4> & vectorExpression){
    return RotationExpression(boost::shared_ptr<Vector2RotationQuaternionExpressionAdapter>(new Vector2RotationQuaternionExpressionAdapter(vectorExpression)));
  }

  ~Vector2RotationQuaternionExpressionAdapter() override;
 protected:
  Vector2RotationQuaternionExpressionAdapter(const VectorExpression<4> & vectorExpression);
  Eigen::Matrix3d toRotationMatrixImplementation() const override;
  void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
  void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;
 private:
  Eigen::MatrixXd getMatrixToLieAlgebra() const;
  boost::shared_ptr<VectorExpressionNode<4> > _root;
  Eigen::Matrix3d _C;
};

} /* namespace backend */
} /* namespace aslam */
#endif /* VECTOR2ROTATIONQUATERNIONEXPRESSIONADAPTER_H_ */
