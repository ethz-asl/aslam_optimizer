#include <Eigen/Geometry>
#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <aslam/backend/KinematicChain.hpp>
#include <aslam/backend/EuclideanPoint.hpp>
#include <aslam/backend/EuclideanExpression.hpp>
#include <aslam/backend/test/ExpressionTests.hpp>


using namespace aslam::backend;

Eigen::Vector3d Zero = Eigen::Vector3d::Zero();
Eigen::Matrix3d Identity = Eigen::Matrix3d::Identity();
Eigen::Vector3d Ones = Eigen::Vector3d::Ones();
Eigen::Matrix3d MinusIdentity = -Eigen::Matrix3d::Identity();
Eigen::Vector3d X = Eigen::Vector3d::UnitX();
Eigen::Vector3d Y = Eigen::Vector3d::UnitY();
Eigen::Vector3d Z = Eigen::Vector3d::UnitZ();

TEST(KinematicChainTestSuites, testTheoreticallyOneFrame) {
  {
    CoordinateFrame A = CoordinateFrame(RotationExpression());
    std::string msg = "Testing all nothing.";

    sm::eigen::assertEqual(A.getPG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getVG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getR_G_L().toRotationMatrix(), Identity, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getOmegaG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAlphaG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
  }
  {
    CoordinateFrame A(MinusIdentity, Ones, Ones, Ones, Ones, Ones);
    std::string msg = "Testing all one.";

    sm::eigen::assertEqual(A.getPG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getVG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getR_G_L().toRotationMatrix(), MinusIdentity, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getOmegaG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAlphaG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
  }
}
TEST(KinematicChainTestSuites, testTheoreticallyTwoFrames) {
  {
    CoordinateFrame B(Identity, Ones, Ones, Ones, Ones, Ones);
    CoordinateFrame A(B, Identity, Ones, Ones, Ones, Ones, Ones);
    std::string msg = "Testing all one.";

    sm::eigen::assertEqual(A.getPG().toValue(), Ones * 2, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getVG().toValue(), Ones * 2, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAG().toValue(), Ones * 2, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getR_G_L().toRotationMatrix(), Identity, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getOmegaG().toValue(), Ones * 2, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAlphaG().toValue(), Ones * 2, SM_SOURCE_FILE_POS, msg);
  }

  {
    CoordinateFrame B(MinusIdentity, Ones, Ones, Ones, Ones, Ones);
    CoordinateFrame A(B, MinusIdentity, Ones, Ones, Ones, Ones, Ones);
    std::string msg = "Testing all one or -Identity.";

    sm::eigen::assertEqual(A.getPG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getVG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getR_G_L().toRotationMatrix(), Identity, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getOmegaG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAlphaG().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
  }

  {
    CoordinateFrame B(Identity, Ones, X, Ones, Ones, Ones);
    CoordinateFrame A(B, Identity, Y, Zero, Zero, Zero, Zero);
    std::string msg = "Testing simple non trivial example";

    sm::eigen::assertEqual(A.getPG().toValue(), Ones + Y, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getVG().toValue(), Ones + Z, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAG().toValue(), Ones + Ones.cross(Y) + X.cross(X.cross(Y)), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getR_G_L().toRotationMatrix(), Identity, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getOmegaG().toValue(), X, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(A.getAlphaG().toValue(), Ones, SM_SOURCE_FILE_POS, msg);
  }

  {
    EuclideanPoint a_m_mrV(Eigen::Vector3d::Random());
    EuclideanPoint t_m_riV(Eigen::Vector3d::Random());
    EuclideanPoint w_m_mrV(Eigen::Vector3d::Random());
    EuclideanPoint dw_m_mrV(Eigen::Vector3d::Random());
    EuclideanExpression a_m_mr(&a_m_mrV);
    EuclideanExpression t_m_ri(&t_m_riV);
    EuclideanExpression w_m_mr(&w_m_mrV);
    EuclideanExpression dw_m_mr(&dw_m_mrV);

    CoordinateFrame M(Identity, X, w_m_mr, Y, dw_m_mr, a_m_mr);
    CoordinateFrame I(M, MinusIdentity, t_m_ri);
    std::string msg = "Testing robot - imu example";

//    aG = pp->getR_G_L() * a + pp->getOmegaG().cross(pp->getR_G_L() * v) + pp->getAlphaG().cross(pp->getR_G_L() * p) + pp->getOmegaG().cross(pp->getOmegaG().cross(pp->getR_G_L() * p));

    EuclideanExpression a_m_mi = a_m_mr + dw_m_mr.cross(t_m_ri) + w_m_mr.cross(w_m_mr.cross(t_m_ri));
    sm::eigen::assertEqual(I.getPP().toValue(), t_m_ri.toValue(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(I.getAP().toValue(), Zero, SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(I.getOmegaG().toValue(), w_m_mr.toValue(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(M.getAG().toValue(), a_m_mr.toValue(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(M.getAlphaG().toValue(), dw_m_mr.toValue(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(I.getAlphaG().toValue(), dw_m_mr.toValue(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(I.getAG().toValue(), a_m_mi.toValue(), SM_SOURCE_FILE_POS, msg);

    testExpression(I.getAG(), 4);
    testExpression(a_m_mi, 4);

    Eigen::MatrixXd m1 = evaluateJacobian(I.getAG(), 4, true);
    Eigen::MatrixXd m2 = evaluateJacobian(a_m_mi, 4, false);
    sm::eigen::assertEqual(m1, m2, SM_SOURCE_FILE_POS, msg);
  }


  //TODO add further test including random
  // CoordinateFrame (R_L_P, p, omega, v, alpha, a)
//  CoordinateFrame
//    C, B(C), A(B);

}

TEST(KinematicChainTestSuites, testTheoreticallyThreeFrames) {
  {
    CoordinateFrame A(Identity, Zero, Z, Zero, Zero, Zero);
    CoordinateFrame B(A, Identity, X, (-Z).eval(), Zero, Zero, Zero);
    CoordinateFrame C(B, Identity, (-Y).eval(), Zero, Zero, Zero, Zero);
    std::string msg = "Testing global acceleration with two complementary rotations";

    sm::eigen::assertEqual(B.getAG().toValue(),(-X).eval(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(C.getAG().toValue(), (-X).eval(), SM_SOURCE_FILE_POS, msg);
  }

  {
    CoordinateFrame A(Identity, Zero, Z, Zero, Zero, Zero);
    CoordinateFrame B(A, Identity, X, Zero, Zero, Zero, Zero);
    CoordinateFrame C(B, Identity, X, Zero, Zero, Zero, Zero);
    std::string msg = "Testing global acceleration with fixed second";

    sm::eigen::assertEqual(B.getAG().toValue(),(-X).eval(), SM_SOURCE_FILE_POS, msg);
    sm::eigen::assertEqual(C.getAG().toValue(), (X * -2).eval(), SM_SOURCE_FILE_POS, msg);
  }
}
