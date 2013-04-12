#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <sm/kinematics/rotations.hpp>
#include <Eigen/Geometry>
#include <aslam/backend/QuaternionExpression.hpp>
#include <aslam/backend/VectorExpressionToGenericMatrixTraits.hpp>
#include <aslam/backend/VectorExpression.hpp>
#include <aslam/backend/DesignVariableVector.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>

using namespace aslam::backend;
using namespace aslam::backend::quaternion;

template<typename TExpression>
struct ExpressionNodeFunctor {
  typedef typename TExpression::value_t value_t;
  typedef typename value_t::Scalar scalar_t;
  typedef Eigen::VectorXd input_t;
  typedef Eigen::MatrixXd jacobian_t;

  ExpressionNodeFunctor(TExpression dv, JacobianContainer & jc)
      : _expression(dv),
        _jc(jc) {
  }

  input_t update(const input_t & x, int c, scalar_t delta) {
    input_t xnew = x;
    xnew[c] += delta;
    return xnew;
  }

  TExpression _expression;
  JacobianContainer & _jc;

  Eigen::VectorXd operator()(const Eigen::VectorXd & dr) {
    int offset = 0;
    for (size_t i = 0; i < _jc.numDesignVariables(); i++) {
      DesignVariable * d = _jc.designVariable(i);
      d->update((const double *) &dr[offset], d->minimalDimensions());
      offset += d->minimalDimensions();
    }

    auto p = _expression.evaluate();

    for (size_t i = 0; i < _jc.numDesignVariables(); i++) {
      DesignVariable * d = _jc.designVariable(i);
      d->revertUpdate();
    }

    return p;
  }
};

template<typename TExpression>
void testJacobian(TExpression expression, int rows = TExpression::value_t::RowsAtCompileTime) {
  typedef ExpressionNodeFunctor<TExpression> functor_t;

  /// Discern the size of the jacobian container
  expression.evaluate();
  JacobianContainer Jc(rows);
  JacobianContainer Jccr(rows);
  expression.evaluateJacobians(Jc);
  expression.evaluateJacobians(Jccr, Eigen::MatrixXd::Identity(rows, rows));

  functor_t functor(expression, Jc);
  sm::eigen::NumericalDiff<ExpressionNodeFunctor<TExpression> > numdiff(functor);

  Eigen::VectorXd dp(Jc.cols());
  dp.setZero();
  Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp);
  sm::eigen::assertNear(Jc.asSparseMatrix(), Jest, 1e-6, SM_SOURCE_FILE_POS, "Testing the Jacobian");
  sm::eigen::assertNear(Jccr.asSparseMatrix(), Jest, 1e-6, SM_SOURCE_FILE_POS, "Testing the Jacobian");
}

template<typename TScalar, enum QuaternionMode EMode>
void testQuaternionBasics() {
  typedef QuaternionExpression<TScalar, EMode> QE;
  typedef UnitQuaternionExpression<TScalar, EMode> UQE;
  typedef aslam::backend::quaternion::internal::EigenQuaternionCalculator<TScalar, EMode> qcalc;

  const int VEC_ROWS = 4;
  typename QE::vector_t val = QE::value_t::Random()
  // {1, 1, 0, 0},
      , val2 = QE::value_t::Random(), valUnit = QE::value_t::Random();
  valUnit /= valUnit.norm();

  QE qExp(val), qExp2(val2);
  UQE qExpUnit(valUnit), ImaginaryBases[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, one = { 1, 0, 0, 0 };

  SCOPED_TRACE("");
  sm::eigen::assertNear(qExp.evaluate(), val, 1e-14, SM_SOURCE_FILE_POS, "Testing evaluation fits initialization.");

  for (int i = 0; i < 3; i++)
    sm::eigen::assertNear((ImaginaryBases[i] * ImaginaryBases[(i + 1) % 3]).evaluate(), ImaginaryBases[(i + 2) % 3].evaluate() * (QE::IsTraditionalMultOrder ? 1 : -1), 1e-14, SM_SOURCE_FILE_POS, "Testing first Hamiltonian equation.");

  sm::eigen::assertNear((qExp + qExp.conjugate()).evaluate(), 2 * val(aslam::backend::quaternion::internal::realIsFirst(EMode) ? 0 : 3) * one.evaluate(), 1e-14, SM_SOURCE_FILE_POS, "Testing an important conjugate equality.");
  sm::eigen::assertNear((qExp * qExp.inverse()).evaluate(), one.evaluate(), 1e-14, SM_SOURCE_FILE_POS, "Testing the important inverse equality.");
  sm::eigen::assertNear((qExpUnit * qExpUnit.inverse()).evaluate(), one.evaluate(), 1e-14, SM_SOURCE_FILE_POS, "Testing the important inverse equality.");

  if (EMode == QuaternionMode::LAST_IS_REAL_AND_OPPOSITE_MULT_ORDER) {
    using namespace sm::kinematics;
    sm::eigen::assertNear((qExp * qExp2).evaluate(), qplus(val, val2), 1e-14, SM_SOURCE_FILE_POS, "Testing conformance with qplus implementation.");
    const double n = val.dot(val);
    sm::eigen::assertNear(qExp.inverse().evaluate(), quatInv(val) / n, 1e-14, SM_SOURCE_FILE_POS, "Testing conformance with quatInv implementation.");
    sm::eigen::assertNear(qExpUnit.inverse().evaluate(), quatInv(valUnit), 1e-14, SM_SOURCE_FILE_POS, "Testing conformance with quatInv implementation.");
  }

  Eigen::Vector3d vec3d = Eigen::Vector3d::Random();
  GenericMatrixExpression<3, 1> vec3dGV(vec3d);

  sm::eigen::assertNear(qExpUnit.rotate3Vector(vec3dGV).evaluate(), qcalc::getImagPart(qcalc::quatMult(qcalc::quatMult(valUnit, vec3d), qcalc::invert(valUnit))), 1e-14, SM_SOURCE_FILE_POS, "Testing conformance with quaternion rotation implementation.");

  DesignVariableVector<VEC_ROWS> dvec, dvec2, dvecUnit;
  DesignVariableVector<3> dvec3d;
  dvec.setActive(true);
  dvec.setBlockIndex(1);
  dvec.setParameters(val);
  dvec2.setActive(true);
  dvec2.setBlockIndex(2);
  dvec2.setParameters(val2);
  dvecUnit.setActive(true);
  dvecUnit.setBlockIndex(3);
  dvecUnit.setParameters(valUnit);
  dvec3d.setActive(true);
  dvec3d.setBlockIndex(4);
  dvec3d.setParameters(vec3d);
  QE qDVarExp(convertToGME(VectorExpression<VEC_ROWS>(&dvec)));
  QE qDVarExp2(convertToGME(VectorExpression<VEC_ROWS>(&dvec2)));
  UQE qDVarUnitExp(convertToGME(VectorExpression<VEC_ROWS>(&dvecUnit)));
  GenericMatrixExpression<3, 1> vec3dDGV(convertToGME(VectorExpression<3>(&dvec3d)));

  {
    testJacobian(qDVarExp);
    testJacobian(qDVarExp.inverse());
    testJacobian(qDVarUnitExp.inverse());
    testJacobian(qDVarExp.conjugate());
    testJacobian(qDVarExp * qExp2);
    testJacobian(qExp - qDVarExp);
    testJacobian(qDVarExp * qDVarExp2);
    testJacobian(qDVarExp2 - qDVarExp);
    testJacobian(qDVarUnitExp.rotate3Vector(vec3dGV));
    testJacobian(qDVarUnitExp.rotate3Vector(vec3dDGV));
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
TESTBASIC_(double, MODE)
//TESTBASIC_(float, MODE)

TESTBASIC(FIRST_IS_REAL_AND_TRADITIONAL_MULT_ORDER)
TESTBASIC(FIRST_IS_REAL_AND_OPPOSITE_MULT_ORDER)
TESTBASIC(LAST_IS_REAL_AND_TRADITIONAL_MULT_ORDER)
TESTBASIC(LAST_IS_REAL_AND_OPPOSITE_MULT_ORDER)

