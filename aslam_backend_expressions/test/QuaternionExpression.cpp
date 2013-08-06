#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <sm/kinematics/rotations.hpp>
#include <Eigen/Geometry>
#include <aslam/backend/QuaternionExpression.hpp>
#include <aslam/backend/GenericMatrixExpression.hpp>
#include <aslam/backend/DesignVariableGenericVector.hpp>
#include <aslam/backend/DesignVariableUnitQuaternion.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>
#include <aslam/backend/test/ExpressionTests.hpp>

using namespace aslam::backend;
using namespace aslam::backend::quaternion;

template<typename Primitve_> constexpr double getTolerance(){
  return 1e-14;
}

template<> constexpr double getTolerance<float>(){
  return 1e-5;
}

#define testJacobian(exp) { SCOPED_TRACE("testJacobian(" #exp ")"); aslam::backend::testJacobian(exp); }

template<typename TScalar, enum QuaternionMode EMode>
void testQuaternionBasics() {
  const double tolerance = getTolerance<TScalar>();
  typedef QuaternionExpression<TScalar, EMode> QE;
  typedef UnitQuaternionExpression<TScalar, EMode> UQE;
  typedef aslam::backend::quaternion::internal::EigenQuaternionCalculator<TScalar, EMode> qcalc;

  const int VEC_ROWS = 4;
  typename QE::vector_t val = QE::value_t::Random()
  // {1, 1, 0, 0},
      , val2 = QE::value_t::Random(), valUnit = QE::value_t::Random(), valUnit2 = QE::value_t::Random();
  valUnit /= valUnit.norm();
  valUnit.setZero();
  valUnit[quaternion::internal::isRealFirst(EMode) ? 0 : 3] = 1;
  valUnit2 /= valUnit2.norm();

  QE qExp(val), qExp2(val2);
  UQE qExpUnit(valUnit), ImaginaryBases[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, one = { 1, 0, 0, 0 };

  sm::eigen::assertEqual(qExp.evaluate(), val, SM_SOURCE_FILE_POS, "Testing evaluation fits initialization.");

  for (int i = 0; i < 3; i++)
    sm::eigen::assertNear((ImaginaryBases[i] * ImaginaryBases[(i + 1) % 3]).evaluate(), ImaginaryBases[(i + 2) % 3].evaluate() * (QE::IsTraditionalMultOrder ? 1 : -1), tolerance, SM_SOURCE_FILE_POS, "Testing first Hamiltonian equation.");

  sm::eigen::assertNear((qExp + qExp.conjugate()).evaluate(), 2 * val(aslam::backend::quaternion::internal::isRealFirst(EMode) ? 0 : 3) * one.evaluate(), tolerance, SM_SOURCE_FILE_POS, "Testing an important conjugate equality.");
  sm::eigen::assertNear((qExp * qExp.inverse()).evaluate(), one.evaluate(), tolerance, SM_SOURCE_FILE_POS, "Testing the important inverse equality.");
  sm::eigen::assertNear((qExpUnit * qExpUnit.inverse()).evaluate(), one.evaluate(), tolerance, SM_SOURCE_FILE_POS, "Testing the important inverse equality.");

  if (EMode == QuaternionMode::LAST_IS_REAL_AND_OPPOSITE_MULT_ORDER) {
    using namespace sm::kinematics;
    sm::eigen::assertNear((qExp * qExp2).evaluate(), qplus(val.template cast<double>(), val2.template cast<double>()).template cast<TScalar>(), tolerance, SM_SOURCE_FILE_POS, "Testing conformance with qplus implementation.");
    const double n = val.dot(val);
    sm::eigen::assertNear(qExp.inverse().evaluate(), quatInv(val.template cast<double>()).template cast<TScalar>() / n, tolerance, SM_SOURCE_FILE_POS, "Testing conformance with quatInv implementation.");
    sm::eigen::assertNear(qExpUnit.inverse().evaluate(), quatInv(valUnit.template cast<double>()).template cast<TScalar>(), tolerance, SM_SOURCE_FILE_POS, "Testing conformance with quatInv implementation.");
  }

  Eigen::Matrix<TScalar, 3, 1> vec3d;
  vec3d.setRandom();
  GenericMatrixExpression<3, 1, TScalar> vec3dGV(vec3d);

  sm::eigen::assertNear(qExpUnit.rotate3Vector(vec3dGV).evaluate(), qcalc::getImagPart(qcalc::quatMult(qcalc::quatMult(valUnit, vec3d), qcalc::invert(valUnit))), tolerance, SM_SOURCE_FILE_POS, "Testing conformance with quaternion rotation implementation.");

  DesignVariableGenericVector<VEC_ROWS, TScalar> dvec, dvec2;
  DesignVariableUnitQuaternion<TScalar, EMode> dvecUnit, dvecUnit2;
  DesignVariableGenericVector<3, TScalar> dvec3d;
  dvec.setActive(true);
  dvec.setBlockIndex(1);
  dvec.setParameters(val);
  dvec2.setActive(true);
  dvec2.setBlockIndex(2);
  dvec2.setParameters(val2);
  dvecUnit.setActive(true);
  dvecUnit.setBlockIndex(3);
  dvecUnit.setParameters(valUnit);
  dvecUnit2.setActive(true);
  dvecUnit2.setBlockIndex(4);
  dvecUnit2.setParameters(valUnit2);
  dvec3d.setActive(true);
  dvec3d.setBlockIndex(4);
  dvec3d.setParameters(vec3d);
  QE qDVarExp(&dvec);
  QE qDVarExp2(&dvec2);
  UnitQuaternionExpression<TScalar, EMode, decltype(dvecUnit)> qDVarUnitExp(&dvecUnit);
  UQE qDVarUnitExp2(&dvecUnit2);
  GenericMatrixExpression<3, 1, TScalar> dVarVec3d(&dvec3d);

  {
    testJacobian(qDVarUnitExp);
    testJacobian(qDVarExp);
    testJacobian(qDVarExp.inverse());
    testJacobian(qDVarUnitExp.inverse());
    testJacobian(qDVarExp.conjugate());
    testJacobian(qDVarExp * qExp2);
    testJacobian(qExp - qDVarExp);
    testJacobian(qDVarExp * qDVarExp2);
    testJacobian(qDVarExp2 - qDVarExp);
    testJacobian(qDVarUnitExp.rotate3Vector(vec3dGV));
    testJacobian(qDVarUnitExp.rotate3Vector(dVarVec3d));
    testJacobian(qDVarUnitExp.geoExp(vec3dGV));
    testJacobian(qDVarUnitExp.geoExp(dVarVec3d));
    testJacobian(qDVarUnitExp.geoLog(qDVarUnitExp2));
  }
}

#define TESTBASIC_(SCALAR, MODE) \
TEST(QuaternionExpressionNodeTestSuites, testQuaternionBasic_##SCALAR##_##MODE) { \
  try { \
    testQuaternionBasics<SCALAR, QuaternionMode::MODE>(); \
  }\
  catch(std::exception const & e)\
  {\
    FAIL() << e.what();\
  }\
}

#define TESTBASIC(MODE)\
TESTBASIC_(float, MODE)\
TESTBASIC_(double, MODE)

TESTBASIC(FIRST_IS_REAL_AND_TRADITIONAL_MULT_ORDER)
TESTBASIC(FIRST_IS_REAL_AND_OPPOSITE_MULT_ORDER)
TESTBASIC(LAST_IS_REAL_AND_TRADITIONAL_MULT_ORDER)
TESTBASIC(LAST_IS_REAL_AND_OPPOSITE_MULT_ORDER)

