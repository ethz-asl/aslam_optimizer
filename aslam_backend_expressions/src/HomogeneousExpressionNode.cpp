#include <aslam/backend/HomogeneousExpressionNode.hpp>
#include <sm/kinematics/transformations.hpp>

namespace aslam {
  namespace backend {
    
    HomogeneousExpressionNode::HomogeneousExpressionNode()
    {

    }

    HomogeneousExpressionNode::~HomogeneousExpressionNode()
    {

    }


    /// \brief Evaluate the homogeneous matrix.
    Eigen::Vector4d HomogeneousExpressionNode::toHomogeneous()
    {
      return toHomogeneousImplementation();
    }

      
    /// \brief Evaluate the Jacobians
    void HomogeneousExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const
    {
      evaluateJacobiansImplementation(outJacobians);
    }
   
    
    /// \brief Evaluate the Jacobians and apply the chain rule.
    void HomogeneousExpressionNode::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      evaluateJacobiansImplementation(outJacobians, applyChainRule);
    }

    void HomogeneousExpressionNode::getDesignVariables(DesignVariable::set_t & designVariables) const
    {
      getDesignVariablesImplementation(designVariables);
    }




    HomogeneousExpressionNodeMultiply::HomogeneousExpressionNodeMultiply(boost::shared_ptr<TransformationExpressionNode> lhs, 
								     boost::shared_ptr<HomogeneousExpressionNode> rhs) :
      _lhs(lhs), _rhs(rhs)
    {
      _T_lhs = _lhs->toTransformationMatrix();
      _p_rhs = _rhs->toHomogeneous();
    }

    HomogeneousExpressionNodeMultiply::~HomogeneousExpressionNodeMultiply()
    {
      
    }


    
    Eigen::Vector4d HomogeneousExpressionNodeMultiply::toHomogeneousImplementation()
    {
      _T_lhs = _lhs->toTransformationMatrix();
      _p_rhs = _rhs->toHomogeneous();

      return _T_lhs * _p_rhs;
    }

    void HomogeneousExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, sm::kinematics::boxMinus(_T_lhs * _p_rhs));
      _rhs->evaluateJacobians(outJacobians, _T_lhs);
    }

    void HomogeneousExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, applyChainRule * sm::kinematics::boxMinus(_T_lhs * _p_rhs));
      _rhs->evaluateJacobians(outJacobians, applyChainRule * _T_lhs);

    }

    void HomogeneousExpressionNodeMultiply::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }




      HomogeneousExpressionNodeConstant::HomogeneousExpressionNodeConstant(const Eigen::Vector4d & p) : _p(p){}
      HomogeneousExpressionNodeConstant::~HomogeneousExpressionNodeConstant(){}


      Eigen::Vector4d HomogeneousExpressionNodeConstant::toHomogeneousImplementation(){ return _p; }
      void HomogeneousExpressionNodeConstant::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const{}
      void HomogeneousExpressionNodeConstant::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const{}
      void HomogeneousExpressionNodeConstant::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const{}

      



  } // namespace backend
} // namespace aslam
