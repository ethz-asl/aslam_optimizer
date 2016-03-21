#include <aslam/backend/TransformationBasic.hpp>
#include <aslam/backend/RotationExpressionNode.hpp>
#include <aslam/backend/EuclideanExpressionNode.hpp>
#include <sm/kinematics/rotations.hpp>

namespace aslam {
  namespace backend {
    TransformationBasic::TransformationBasic(RotationExpression C_0_1, EuclideanExpression t_0_1_0) :
      _rotation(C_0_1.root()), _translation(t_0_1_0.root())
    {
    }

    TransformationBasic::~TransformationBasic()
    {
    }

    Eigen::Matrix4d TransformationBasic::toTransformationMatrixImplementation()
    {
      Eigen::Matrix4d T;
      T.setIdentity();
      if(_rotation)
        T.topLeftCorner<3,3>() = _rotation->toRotationMatrix();
      if(_translation)
        T.topRightCorner<3,1>() = _translation->toEuclidean();
      return T;
    }

    void TransformationBasic::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      //S(x) = [ 1    - r^\times S(theta) ]
      //       [ 0         S(theta)       ]
      // Eigen::Matrix3d C = _rotation->toRotationMatrix();
      if(_rotation){
        Eigen::Matrix<double,6,3> crRotation;
        if(_translation){
          crRotation.topLeftCorner<3,3>() = -sm::kinematics::crossMx(_translation->toEuclidean());
        } else {
          crRotation.topLeftCorner<3,3>().setZero();
        }
        _rotation->evaluateJacobians(outJacobians, crRotation);
      }

      if(_translation) {
        Eigen::Matrix<double,6,3> crTranslation;
        crTranslation.topLeftCorner<3,3>().setIdentity();
        crTranslation.bottomLeftCorner<3,3>().setZero();
        _translation->evaluateJacobians(outJacobians, crTranslation);
      }
    }

    void TransformationBasic::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      //S(x) = [ 1    - r^\times S(theta) ]
      //       [ 0         S(theta)       ]
      // Eigen::Matrix3d C = _rotation->toRotationMatrix();
      if(_rotation){
        Eigen::Matrix<double,6,3> crRotation;
        if(_translation){
          crRotation.topLeftCorner<3,3>() = -sm::kinematics::crossMx(_translation->toEuclidean());
        } else {
          crRotation.topLeftCorner<3,3>().setZero();
        }
        crRotation.bottomLeftCorner<3,3>().setIdentity();
        _rotation->evaluateJacobians(outJacobians, applyChainRule * crRotation);
      }
      
      if(_translation) {
        Eigen::Matrix<double,6,3> crTranslation;
        crTranslation.topLeftCorner<3,3>().setIdentity();
        crTranslation.bottomLeftCorner<3,3>().setZero();
        _translation->evaluateJacobians(outJacobians, applyChainRule * crTranslation);
      }
    }

    void TransformationBasic::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      if(_rotation)
        _rotation->getDesignVariables(designVariables);
      if(_translation)
        _translation->getDesignVariables(designVariables);
    }



    TransformationExpression TransformationBasic::toExpression()
    {
      return TransformationExpression(this);
    }

  } // namespace backend
} // namespace aslam
