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

namespace internal {
template<typename TExpression>
struct ExpressionNodeFunctor {
  typedef typename TExpression::value_t value_t;
  typedef typename value_t::Scalar scalar_t;
  typedef Eigen::VectorXd input_t;
  typedef Eigen::MatrixXd jacobian_t;

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
} // namespace internal


template<typename TExpression>
void testJacobian(TExpression expression, bool printResult = false) {
  typedef internal::ExpressionNodeFunctor<TExpression> functor_t;
  auto val = expression.evaluate();
  int rows = val.rows();

  JacobianContainer Jc(rows);
  JacobianContainer Jccr(rows);
  expression.evaluateJacobians(Jc);
  expression.evaluateJacobians(Jccr, Eigen::MatrixXd::Identity(rows, rows));

  functor_t functor(expression, Jc);
  sm::eigen::NumericalDiff<internal::ExpressionNodeFunctor<TExpression> > numdiff(functor);

  Eigen::VectorXd dp(Jc.cols());
  dp.setZero();
  Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp);
  auto JcM = Jc.asDenseMatrix();
  sm::eigen::assertNear(Jest, JcM, 1e-6, SM_SOURCE_FILE_POS, "Testing the Jacobian with finite differences");
  sm::eigen::assertEqual(JcM, Jccr.asDenseMatrix(), SM_SOURCE_FILE_POS, "Testing whether appending identity changes nothing.");

  if(printResult){
    std::cout << "Jest=\n" << Jest << std::endl; // XXX: debug output of Jest
    std::cout << "Jc=\n" << JcM << std::endl; // XXX: debug output of Jest
    std::cout << "Jccr=\n" << Jccr.asDenseMatrix() << std::endl; // XXX: debug output of Jest
  }
}

}
}
#endif /* EXPRESSIONTESTS_HPP_ */
