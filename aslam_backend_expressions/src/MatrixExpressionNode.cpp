#include <aslam/backend/MatrixExpressionNode.hpp>

namespace aslam {
  namespace backend {
    
    ////////////////////////////////////////////
    // RotationExpressionNode: The Super Class
    ////////////////////////////////////////////
    MatrixExpressionNode::MatrixExpressionNode(){}

    MatrixExpressionNode::~MatrixExpressionNode(){}

    Eigen::Matrix3d MatrixExpressionNode::toFullMatrix(){ return toFullMatrixImplementation(); }

    void MatrixExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const{
      evaluateJacobiansImplementation(outJacobians);
    }      

    void MatrixExpressionNode::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const{
      evaluateJacobiansImplementation(outJacobians, applyChainRule);
    }
      
    void MatrixExpressionNode::getDesignVariables(DesignVariable::set_t & designVariables) const
    {
      getDesignVariablesImplementation(designVariables);
    }

  } // namespace backend
} // namespace aslam
