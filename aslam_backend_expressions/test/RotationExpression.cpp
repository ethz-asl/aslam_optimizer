#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <aslam/backend/RotationExpression.hpp>
#include <sm/kinematics/rotations.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>
#include <aslam/backend/RotationQuaternion.hpp>


using namespace aslam::backend;
using namespace sm::kinematics;

struct RotationExpressionNodeFunctor
{
  typedef Eigen::Vector3d value_t;
  typedef value_t::Scalar scalar_t;
  typedef Eigen::VectorXd input_t;
  typedef Eigen::MatrixXd jacobian_t;

  
  RotationExpressionNodeFunctor(RotationExpression dv, Eigen::Vector3d p) :
    _p(p), _dv(dv) {}

  input_t update(const input_t & x, int c, scalar_t delta) { input_t xnew = x; xnew[c] += delta; return xnew; }

  Eigen::VectorXd _p;
  RotationExpression _dv;

  Eigen::VectorXd operator()(const Eigen::VectorXd & dr)
  {
    
    Eigen::Matrix3d C = _dv.toRotationMatrix();
    JacobianContainer J(3);
    _dv.evaluateJacobians(J);

    int offset = 0;
    for(size_t i = 0; i < J.numDesignVariables(); i++)
      {
	DesignVariable * d = J.designVariable(i);
	d->update(&dr[offset],d->minimalDimensions());
	offset += d->minimalDimensions();
      }

    C = _dv.toRotationMatrix();
    
 
    for(size_t i = 0; i < J.numDesignVariables(); i++)
      {
	DesignVariable * d = J.designVariable(i);
	d->revertUpdate();
      }

    return C*_p;
   
  }
};


void testJacobian(RotationExpression dv)
{
  Eigen::Vector3d p;
  p.setRandom();
  RotationExpressionNodeFunctor functor(dv,p);
  
  sm::eigen::NumericalDiff<RotationExpressionNodeFunctor> numdiff(functor);
  
  /// Discern the size of the jacobian container
  Eigen::Matrix3d C = dv.toRotationMatrix();
  JacobianContainer Jc(3);
  dv.evaluateJacobians(Jc);
  Eigen::Matrix3d Cp_cross = sm::kinematics::crossMx(C*p);
  Jc.applyChainRule(Cp_cross);
 
  Eigen::VectorXd dp(Jc.cols());
  dp.setZero();
  Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp);
 
  sm::eigen::assertNear(Jc.asSparseMatrix(), Jest, 1e-6, SM_SOURCE_FILE_POS, "Testing the quat Jacobian");
}





// Test that the quaternion jacobian matches the finite difference jacobian
TEST(RotationExpressionNodeTestSuites, testQuat)
{
  try 
    {
      using namespace sm::kinematics;
      RotationQuaternion quat(quatRandom());
      quat.setActive(true);
      quat.setBlockIndex(0);
      RotationExpression qr(&quat);
      
      testJacobian(qr);
    }
  catch(std::exception const & e)
    {
      FAIL() << e.what();
    }
}

// Test that the inverse quaternion matches the finite difference.
TEST(RotationExpressionNodeTestSuites, testQuatInverse1)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat(quatRandom());
    quat.setActive(true);
    quat.setBlockIndex(0);
    RotationExpression qr(&quat);

    RotationExpression invqr = qr.inverse();
    
    testJacobian(invqr);
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

// Test that the inverse computes the correct result.
TEST(RotationExpressionNodeTestSuites, testQuatInverse2)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat(quatRandom());
    quat.setActive(true);
    quat.setBlockIndex(0);
    RotationExpression qr(&quat);

    RotationExpression dvInverse = qr.inverse();
    
    Eigen::Matrix3d invQ1 = dvInverse.toRotationMatrix();
    Eigen::Matrix3d invQ2 = qr.toRotationMatrix().transpose();

    sm::eigen::assertNear(invQ1, invQ2, 1e-14, SM_SOURCE_FILE_POS, "Test that the inverse computes the correct result");
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}


TEST(RotationExpressionNodeTestSuites, testQuatMultiply1)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat(quatRandom());
    quat.setActive(true);
    quat.setBlockIndex(0);
    RotationExpression qr(&quat);

    RotationExpression dvMultiply = qr * qr;
    
    testJacobian(dvMultiply);
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}


// Test that the binary multiplication produces the correct result.
TEST(RotationExpressionNodeTestSuites, testQuatMultiply2)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);
    RotationExpression dvMultiply = qr0 * qr1;

    sm::eigen::assertEqual(qr0.toRotationMatrix() * qr1.toRotationMatrix(), dvMultiply.toRotationMatrix(), SM_SOURCE_FILE_POS, "Test multiplications");
    
    testJacobian(dvMultiply);
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}


/// Test that 3x multiplication (q0,q1) q2 produces the correct result
TEST(RotationExpressionNodeTestSuites, testQuatMultiply3a)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);

    RotationQuaternion quat2(quatRandom());
    quat2.setActive(true);
    quat2.setBlockIndex(2);
    RotationExpression qr2(&quat2);

    RotationExpression dvMultiply = qr2 * (qr0 * qr1);
    

    sm::eigen::assertNear(qr2.toRotationMatrix() * qr0.toRotationMatrix() * qr1.toRotationMatrix(), dvMultiply.toRotationMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");
    
    testJacobian(dvMultiply);

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}


/// Test that 3x multiplication (q0,q1) q2 and q0 (q1 q2) produce the same result
 TEST(RotationExpressionNodeTestSuites,stQuatMulty3b)
 {
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);

    RotationQuaternion quat2(quatRandom());
    quat2.setActive(true);
    quat2.setBlockIndex(2);
    RotationExpression qr2(&quat2);

    RotationExpression dvMultiply01 = qr0 * qr1;
   
    // Right multiplication:
    RotationExpression dvMultiply02Right = dvMultiply01 * qr2; 

    RotationExpression dvMultiply12 = qr1 * qr2;    
    RotationExpression dvMultiply02Left = qr0 * dvMultiply12;    

    
    sm::eigen::assertNear( dvMultiply02Left.toRotationMatrix(), dvMultiply02Right.toRotationMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");
    
    JacobianContainer JcLeft(3);
    dvMultiply02Left.evaluateJacobians(JcLeft);

    JacobianContainer JcRight(3);
    dvMultiply02Right.evaluateJacobians(JcRight);
    
    sm::eigen::assertNear( JcLeft.asSparseMatrix(), JcRight.asSparseMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");
    

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

/// Test that 3x multiplication  q0 (q1 q2) produces the correct result
 TEST(RotationExpressionNodeTestSuites,stQuatMulty3c)
 {
  try {
     using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);

    RotationQuaternion quat2(quatRandom());
    quat2.setActive(true);
    quat2.setBlockIndex(2);
    RotationExpression qr2(&quat2);

    RotationExpression dvMultiply = qr0 * qr1;
   
    // Right multiplication:
    RotationExpression dvMultiplyRight = dvMultiply * qr2; 

    sm::eigen::assertNear( qr0.toRotationMatrix() * qr1.toRotationMatrix() * qr2.toRotationMatrix(), dvMultiplyRight.toRotationMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");

    testJacobian(dvMultiplyRight);

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}


/// Test that 3x multiplication  q0 (q1 q2)^-1 produces the correct result
 TEST(RotationExpressionNodeTestSuites,stQuatMultInv1)
 {
  try {
     using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);

    RotationQuaternion quat2(quatRandom());
    quat2.setActive(true);
    quat2.setBlockIndex(2);
    RotationExpression qr2(&quat2);

    RotationExpression dvMultiply01 = qr0 * qr1;
   
    RotationExpression dvInverse10 = dvMultiply01.inverse();


    // Right multiplication:
    RotationExpression dvMultiplyRight210 = qr2 * dvInverse10; 

    sm::eigen::assertNear( qr2.toRotationMatrix() * (qr0.toRotationMatrix() * qr1.toRotationMatrix()).transpose(), dvMultiplyRight210.toRotationMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");

    testJacobian(dvMultiplyRight210);

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

/// Test that 3x multiplication  q0^1 (q1 q2) produces the correct result
 TEST(RotationExpressionNodeTestSuites,stQuatMultInv2)
 {
  try {
     using namespace sm::kinematics;
    RotationQuaternion quat0(quatRandom());
    quat0.setActive(true);
    quat0.setBlockIndex(0);
    RotationExpression qr0(&quat0);

    RotationQuaternion quat1(quatRandom());
    quat1.setActive(true);
    quat1.setBlockIndex(1);
    RotationExpression qr1(&quat1);

    RotationQuaternion quat2(quatRandom());
    quat2.setActive(true);
    quat2.setBlockIndex(2);
    RotationExpression qr2(&quat2);

    RotationExpression dvMultiply01 = qr0 * qr1;
   
    RotationExpression dvInverse2 = qr2.inverse();


    // Right multiplication:
    RotationExpression dvMultiplyRight210 = dvInverse2 * dvMultiply01; 

    sm::eigen::assertNear( qr2.toRotationMatrix().transpose() * (qr0.toRotationMatrix() * qr1.toRotationMatrix()), dvMultiplyRight210.toRotationMatrix(), 1e-14, SM_SOURCE_FILE_POS, "Test multiplications");

    testJacobian(dvMultiplyRight210);

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

//Test that the inverse quaternion matches the finite difference.
TEST(RotationExpressionNodeTestSuites, testQuatInverseTemplate1)
{
  try {
    using namespace sm::kinematics;
    RotationQuaternion quat(quatRandom());
    quat.setActive(true);
    quat.setBlockIndex(0);
    RotationExpression qr(&quat);
    
    RotationExpression qi = qr.inverse();
    testJacobian(qi);
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}
