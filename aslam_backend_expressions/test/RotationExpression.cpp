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
    
    //Eigen::Matrix3d C = _dv.toRotationMatrix();
    JacobianContainer J(3);
    _dv.evaluateJacobians(J);

    int offset = 0;
    for(size_t i = 0; i < J.numDesignVariables(); i++)
      {
		DesignVariable * d = J.designVariable(i);
		d->update(&dr[offset],d->minimalDimensions());
		offset += d->minimalDimensions();
      }

    Eigen::Matrix3d C = _dv.toRotationMatrix();
    
 
    for(size_t i = 0; i < J.numDesignVariables(); i++)
      {
	DesignVariable * d = J.designVariable(i);
	d->revertUpdate();
      }

    return C*_p;
   
  }
};
//
//struct RotationExpressionLogNodeFunctor
//{
//  typedef Eigen::Vector3d value_t;
//  typedef value_t::Scalar scalar_t;
//  typedef Eigen::VectorXd input_t;
//  typedef Eigen::MatrixXd jacobian_t;
//
//
//  RotationExpressionLogNodeFunctor(RotationQuaternion dv) : //, Eigen::Vector4d xHat) :
//   _dv(dv) {}
//
//  input_t update(const input_t & x, int c, scalar_t delta) { input_t xnew = x; xnew[c] += delta; return xnew; }
//
//  RotationQuaternion _dv;
//  Eigen::Vector4d _xHat;
//
//  Eigen::Vector3d operator()(const Eigen::VectorXd & dr)
//  {
//	    std::setprecision(15);
//	  std::cout << "perturbing by: " << std::endl << dr << std::endl;
//
//    //Eigen::Matrix3d C = _dv.toRotationMatrix();
//    JacobianContainer J(3);
//    _dv.evaluateJacobians(J);
//
//    Eigen::MatrixXd params1;
//    _dv.getParameters(params1);
//    Eigen::Vector4d quatParams1;
//    quatParams1(0) = params1(0,0); quatParams1(1) = params1(1,0); quatParams1(2) = params1(2,0); quatParams1(3) = params1(3,0);
//
//    Eigen::Vector3d out1  = sm::kinematics::qlog(quatParams1);
//
//    std::cout << "axis angle before the update: " << std::endl << out1 << std::endl;
//
//    int offset = 0;
//    for(size_t i = 0; i < J.numDesignVariables(); i++)
//      {
//		DesignVariable * d = J.designVariable(i);
//		d->update(&dr[offset],d->minimalDimensions());
//		offset += d->minimalDimensions();
//      }
//
//    Eigen::MatrixXd params;
//    _dv.getParameters(params);
//    Eigen::Vector4d quatParams;
//    quatParams(0) = params(0,0); quatParams(1) = params(1,0); quatParams(2) = params(2,0); quatParams(3) = params(3,0);
//    Eigen::Vector3d out  = sm::kinematics::qlog(quatParams);
//    std::cout << "axis angle after the update: " << std::endl << out << std::endl;
//    std::setprecision(5);
//    for(size_t i = 0; i < J.numDesignVariables(); i++)
//      {
//		DesignVariable * d = J.designVariable(i);
//		d->revertUpdate();
//      }
//
//    return out;
//
//  }
//};

TEST(RotationExpressionNodeTestSuites,testQuatLogJacobian)
{
	try {

	    using namespace sm::kinematics;
	    Eigen::Vector4d initialValue = quatRandom();
	    //RotationQuaternion quat(initialValue);
	    //quat.setActive(true);
	    //quat.setBlockIndex(0);

//	    RotationExpressionLogNodeFunctor functor(quat);
//	    sm::eigen::NumericalDiff<RotationExpressionLogNodeFunctor> numdiff(functor);

	    double eps = 1e-7;

	    Eigen::Vector3d AAInitial = sm::kinematics::qlog(initialValue);

	    std::cout << std::setprecision(15) << "Initial Value: " << std::endl << initialValue << std::endl;
	    std::cout << std::setprecision(15) << "Initial AA: " << std::endl << AAInitial << std::endl;

	    Eigen::MatrixXd Jest = Eigen::MatrixXd(3,4);

    	double dx = 2*eps;

	    for(int c = 0; c < 4; c++)
	    {
	    	Eigen::Vector4d updatedQuat;
	    	updatedQuat = initialValue;
	    	updatedQuat(c) += eps;
	    	//std::cout << "Updated quat: " << std::endl << updatedQuat << std::endl;
	    	Eigen::Vector3d AAUpdatedPlus = sm::kinematics::qlog(updatedQuat);
	    	//std::cout << "AAUpdatedPlus: " << std::endl << AAUpdatedPlus << std::endl;
	    	updatedQuat = initialValue;
	    	updatedQuat(c) -= eps;
	    	Eigen::Vector3d AAUpdatedMinus = sm::kinematics::qlog(updatedQuat);

	    	Eigen::Vector3d diffAA =  AAUpdatedPlus - AAUpdatedMinus;
	    	//std::cout << "Diff AA: " << std::endl << diffAA << std::endl;
	    	Jest(0,c) = diffAA(0) / dx;
	    	Jest(1,c) = diffAA(1) / dx;
	    	Jest(2,c) = diffAA(2) / dx;
	    }

	    Eigen::MatrixXd J = sm::kinematics::quatLogJacobian(initialValue);

		std::cout << "Jest" << std::endl << Jest << std::endl;
		std::cout << "J" << std::endl << J << std::endl;

		sm::eigen::assertNear(J, Jest, 1e-6, SM_SOURCE_FILE_POS, "Testing the quat log Jacobian");

	}
	catch(std::exception const & e)
    {
        FAIL() << e.what();
    }

}


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
  std::cout << "Jc:" << std::endl << Jc.asDenseMatrix() << std::endl;
  std::cout << "C*p:" << std::endl << C*p << std::endl;
  Eigen::Matrix3d Cp_cross = sm::kinematics::crossMx(C*p);
  std::cout << "Cp_cross" << std::endl << Cp_cross << std::endl;
  Jc.applyChainRule(Cp_cross);
  std::cout << "Jc" << std::endl << Jc.asDenseMatrix() << std::endl;
 
  Eigen::VectorXd dp(Jc.cols());
  dp.setZero();
  Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp);
  std::cout << "Jest" << std::endl << Jest << std::endl;
 
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

TEST(RotationExpressionNodeTestSuites, testMinimalDifference)
{
  try {
    using namespace sm::kinematics;
    Eigen::Vector4d initialValue = quatRandom();
    RotationQuaternion quat(initialValue);
    quat.setActive(true);
    quat.setBlockIndex(0);

    Eigen::Vector4d epsQ = quatRandom();
    Eigen::Vector3d eps = sm::kinematics::qlog(epsQ);
    double updateValues[3] = {eps(0), eps(1), eps(2)};
    quat.update(updateValues, 3);

    Eigen::VectorXd diff;
    quat.minimalDifference(initialValue, diff);

    EXPECT_TRUE(eps.isApprox(diff));

  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

TEST(RotationExpressionNodeTestSuites, testMinimalDifferenceAndJacobian)
{
  try {
    using namespace sm::kinematics;
    Eigen::Vector4d quatInitial = quatRandom();
    RotationQuaternion quat(quatInitial);
    quat.setActive(true);
    quat.setBlockIndex(0);

    Eigen::Vector4d updatedQuat = quatRandom();
    Eigen::Vector3d updateAA = sm::kinematics::qlog(updatedQuat);
    double updateValues[3] = {updateAA(0), updateAA(1), updateAA(2)};
    quat.update(updateValues, 3);

    //std::cout << "initial value is " << std::endl << quatInitial << std::endl;

    Eigen::MatrixXd quatHat = Eigen::MatrixXd(4,1);
    quatHat(0,0) = quatInitial(0); quatHat(1,0) = quatInitial(1); quatHat(2,0) = quatInitial(2,0); quatHat(3,0) = quatInitial(3,0);


    Eigen::MatrixXd params;
    quat.getParameters(params);
    Eigen::Vector4d quatBar;
    quatBar(0) = params(0,0); quatBar(1) = params(1,0); quatBar(2) = params(2,0); quatBar(3) = params(3,0);

    Eigen::VectorXd minDist;
    Eigen::MatrixXd M;
    quat.minimalDifferenceAndJacobian(quatInitial, minDist, M);

    // choose small
    Eigen::Vector3d epsV; epsV.setRandom();epsV = epsV * 0.05;//epsV(0) = 0.001; epsV(1) = 0.001; epsV(2) = 0.001;
    double eps[3] = {epsV(0), epsV(1), epsV(2)};
    quat.update(eps, 3);

    Eigen::MatrixXd params2;
    quat.getParameters(params2);
    Eigen::Vector4d quatFinal;
    quatFinal(0) = params2(0,0); quatFinal(1) = params2(1,0); quatFinal(2) = params2(2,0); quatFinal(3) = params2(3,0);

    Eigen::Vector3d realMinDist = sm::kinematics::qlog(sm::kinematics::qplus(quatFinal, sm::kinematics::quatInv(quatInitial)));

    Eigen::Vector3d estMinDist = minDist + M*epsV;

    std::cout << "realMinDist" << std::endl << realMinDist << std::endl;
    std::cout << "estMinDist" << std::endl << estMinDist << std::endl;

    sm::eigen::assertNear(realMinDist, estMinDist, 1e-2, SM_SOURCE_FILE_POS, "Test min difference with jacobian");
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}

