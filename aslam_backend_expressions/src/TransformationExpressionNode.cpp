#include <aslam/backend/TransformationExpressionNode.hpp>
#include <sm/kinematics/transformations.hpp>
#include <aslam/backend/EuclideanExpression.hpp>
#include <aslam/backend/EuclideanExpressionNode.hpp>
#include <aslam/backend/RotationExpression.hpp>
#include <aslam/backend/RotationExpressionNode.hpp>
#include <Eigen/Dense>

namespace aslam {
  namespace backend {
    
    ////////////////////////////////////////////
    // TransformationExpressionNode: The Super Class
    ////////////////////////////////////////////
    TransformationExpressionNode::TransformationExpressionNode(){}

    TransformationExpressionNode::~TransformationExpressionNode(){}

    Eigen::Matrix4d TransformationExpressionNode::toTransformationMatrix(){ return toTransformationMatrixImplementation(); }

    void TransformationExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const
    {
      evaluateJacobiansImplementation(outJacobians);
    }      
      
    void TransformationExpressionNode::getDesignVariables(DesignVariable::set_t & designVariables) const
    {
      getDesignVariablesImplementation(designVariables);
    }


    /////////////////////////////////////////////////
    // TransformationExpressionNodeMultiply: A container for C1 * C2
    /////////////////////////////////////////////////

    TransformationExpressionNodeMultiply::TransformationExpressionNodeMultiply(boost::shared_ptr<TransformationExpressionNode> lhs, boost::shared_ptr<TransformationExpressionNode> rhs):
      _lhs(lhs), _rhs(rhs)
    {
    }

    TransformationExpressionNodeMultiply::~TransformationExpressionNodeMultiply(){}

    void TransformationExpressionNodeMultiply::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }


    Eigen::Matrix4d TransformationExpressionNodeMultiply::toTransformationMatrixImplementation()
    {
      _T_lhs = _lhs->toTransformationMatrix();
      _T_rhs = _rhs->toTransformationMatrix();
      return  _T_lhs * _T_rhs;
    }

    void TransformationExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      Eigen::Matrix<double, 6, 6> m;
      auto C = _T_lhs.topLeftCorner<3, 3>();
      m.topLeftCorner<3, 3>() = C;
      m.topRightCorner<3, 3>().setZero();
      m.bottomRightCorner<3, 3>() = C;
      m.bottomLeftCorner<3, 3>().setZero();

      _rhs->evaluateJacobians(outJacobians, m);

      m.topLeftCorner<3, 3>().setIdentity();
      m.topRightCorner<3, 3>() = sm::kinematics::crossMx(C * _T_rhs.topRightCorner<3, 1>());
      m.bottomRightCorner<3, 3>().setIdentity();

      _lhs->evaluateJacobians(outJacobians, m);
    }

    ////////////////////////////////////////////////////
    /// TransformationExpressionNodeInverse: A container for T^-1
    ////////////////////////////////////////////////////
    
    TransformationExpressionNodeInverse::TransformationExpressionNodeInverse(boost::shared_ptr<TransformationExpressionNode> dvTransformation) : _dvTransformation(dvTransformation)
    {
      _T = _dvTransformation->toTransformationMatrix().inverse();
    }
    
    TransformationExpressionNodeInverse::~TransformationExpressionNodeInverse(){}

    Eigen::Matrix4d TransformationExpressionNodeInverse::toTransformationMatrixImplementation()
    {
      _T = _dvTransformation->toTransformationMatrix().inverse();
      return  _T;
    }

    void TransformationExpressionNodeInverse::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      Eigen::Matrix<double, 6, 6> m;
      auto C = _T.topLeftCorner<3, 3>();
      m.topLeftCorner<3, 3>() = -C;
      m.topRightCorner<3, 3>() = -sm::kinematics::crossMx(_T.topRightCorner(3, 1)) * C;
      m.bottomRightCorner<3, 3>() = -C;
      m.bottomLeftCorner<3, 3>().setZero();

      _dvTransformation->evaluateJacobians(outJacobians, m);
    }

    void TransformationExpressionNodeInverse::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _dvTransformation->getDesignVariables(designVariables);
    }

      TransformationExpressionNodeConstant::TransformationExpressionNodeConstant(const Eigen::Matrix4d & T) : _T(T) {}
      TransformationExpressionNodeConstant::~TransformationExpressionNodeConstant(){}

    
      Eigen::Matrix4d TransformationExpressionNodeConstant::toTransformationMatrixImplementation(){ return _T; }
  void TransformationExpressionNodeConstant::evaluateJacobiansImplementation(JacobianContainer & /* outJacobians */) const{}
  void TransformationExpressionNodeConstant::getDesignVariablesImplementation(DesignVariable::set_t & /* designVariables */) const{}

  RotationExpression aslam::backend::TransformationExpressionNode::toRotationExpression(const boost::shared_ptr<TransformationExpressionNode>& thisShared) const {
    assert(thisShared.get() == this);
    return boost::shared_ptr< RotationExpressionNode >( new RotationExpressionNodeTransformation( thisShared ) );
  }

  EuclideanExpression aslam::backend::TransformationExpressionNode::toEuclideanExpression(const boost::shared_ptr<TransformationExpressionNode>& thisShared) const {
    assert(thisShared.get() == this);
    return boost::shared_ptr< EuclideanExpressionNode >( new EuclideanExpressionNodeTranslation( thisShared ) );
  }

  } // namespace backend
} // namespace aslam
