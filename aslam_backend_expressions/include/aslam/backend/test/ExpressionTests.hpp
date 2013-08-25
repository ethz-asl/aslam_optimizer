/*
 * ExpressionTests.hpp
 *
 *  Created on: Jul 16, 2013
 *      Author: hannes
 */

#ifndef EXPRESSIONTESTS_HPP_
#define EXPRESSIONTESTS_HPP_

#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <aslam/backend/JacobianContainer.hpp>

namespace aslam {
namespace backend {
namespace test {

template <typename TExpression>
struct ExpressionValueTraits {
  typedef typename TExpression::value_t value_t;
};

template <typename TExpression>
struct ExpressionTraits {
  typedef typename ExpressionValueTraits<TExpression>::value_t::Scalar scalar_t;
  static scalar_t defaultTolerance() {
    return sqrt(std::numeric_limits<scalar_t>::epsilon()) * 1E2;
  }
  static scalar_t defaulEps() {
    return sqrt(std::numeric_limits<scalar_t>::epsilon()) * 10;
  }
};

template<typename TExpression>
class ExpressionTester {
 public:
  static void testJacobian(TExpression & expression, bool printResult = false, double tolerance = ExpressionTraits<TExpression>::defaultTolerance(), double eps = ExpressionTraits<TExpression>::defaultEps());

 private:
  struct ExpressionNodeFunctor {
    typedef typename TExpression::value_t value_t;
    typedef typename value_t::Scalar scalar_t;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> input_t;
    typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> jacobian_t;

    ExpressionNodeFunctor(TExpression & dv, JacobianContainer & jc)
        : _expression(dv),
          _jc(jc) {
    }

    input_t update(const input_t & x, int c, scalar_t delta) {
      input_t xnew = x;
      xnew[c] += delta;
      return xnew;
    }

    TExpression & _expression;
    JacobianContainer & _jc;

    value_t operator()(const input_t & dr) {
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
};

template<typename TExpression>
void ExpressionTester<TExpression>::testJacobian(TExpression & expression, bool printResult, double tolerance, double eps) {
  auto val = expression.evaluate();
  int rows = val.rows();

  JacobianContainer Jc(rows);
  JacobianContainer Jccr(rows);
  expression.evaluateJacobians(Jc);
  expression.evaluateJacobians(Jccr, Eigen::MatrixXd::Identity(rows, rows));

  int jacobianCols = Jc.cols();
  DesignVariable::set_t designVariables;
  expression.getDesignVariables(designVariables);
  int minimalDimensionSum = 0;
  for(DesignVariable * dp : designVariables){
    minimalDimensionSum += dp->minimalDimensions();
  }
  ASSERT_EQ(jacobianCols, minimalDimensionSum);
  sm::eigen::NumericalDiff<ExpressionNodeFunctor> numdiff(ExpressionNodeFunctor(expression, Jc), eps);
  typename ExpressionNodeFunctor::input_t dp(jacobianCols);

  dp.setZero();
  Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp).template cast<double>();
  auto JcM = Jc.asDenseMatrix();
  sm::eigen::assertNear(Jest, JcM, tolerance, SM_SOURCE_FILE_POS, "Testing the Jacobian with finite differences");
  sm::eigen::assertEqual(JcM, Jccr.asDenseMatrix(), SM_SOURCE_FILE_POS, "Testing whether appending identity changes nothing.");

  if(printResult){
    std::cout << "Jest=\n" << Jest << std::endl;
    std::cout << "Jc=\n" << JcM << std::endl;
    std::cout << "Jccr=\n" << Jccr.asDenseMatrix() << std::endl;
  }
}

} // namespace test

template<typename TExpression>
inline void testJacobian(TExpression expression, bool printResult = false, double tolerance = test::ExpressionTraits<TExpression>::defaultTolerance(), double eps = test::ExpressionTraits<TExpression>::defaulEps()) {
  test::ExpressionTester<TExpression>::testJacobian(expression, printResult, tolerance, eps);
}

}
}
#endif /* EXPRESSIONTESTS_HPP_ */
