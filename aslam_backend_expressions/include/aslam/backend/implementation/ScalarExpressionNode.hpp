/*
 * ScalarExpressionNode.hpp
 *
 *  Created on: 29.03.2016
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_IMPLEMENTATION_SCALAREXPRESSIONNODE_HPP_
#define INCLUDE_ASLAM_BACKEND_IMPLEMENTATION_SCALAREXPRESSIONNODE_HPP_

#include <cmath>

namespace aslam {
namespace backend {


/// \brief Evaluate the scalar matrix.
double ScalarExpressionNode::toScalar() const
{
  return evaluateImplementation();
}

/// \brief Evaluate the Jacobians
void ScalarExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const
{
  evaluateJacobiansImplementation(outJacobians);
}

double ScalarExpressionNodeMultiply::evaluateImplementation() const
{
  return _lhs->toScalar() * _rhs->toScalar();
}

void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  _lhs->evaluateJacobians(outJacobians.apply(_rhs->toScalar()));
  _rhs->evaluateJacobians(outJacobians.apply(_lhs->toScalar()));
}

double ScalarExpressionNodeDivide::evaluateImplementation() const
{
  return _lhs->toScalar() / _rhs->toScalar();
}

void ScalarExpressionNodeDivide::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  const auto rhs_rec = 1./_rhs->toScalar();
  const auto R = -_lhs->toScalar() * rhs_rec * rhs_rec;
  _lhs->evaluateJacobians(outJacobians.apply(rhs_rec));
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeNegated::evaluateImplementation() const
{
  return -_rhs->toScalar();
}

void ScalarExpressionNodeNegated::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  _rhs->evaluateJacobians(outJacobians.apply(-1.0));
}

double ScalarExpressionNodeAdd::evaluateImplementation() const
{
  return _lhs->toScalar() + _multiplyRhs * _rhs->toScalar();
}

void ScalarExpressionNodeAdd::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  _lhs->evaluateJacobians(outJacobians);
  _rhs->evaluateJacobians(outJacobians.apply(_multiplyRhs));
}

double ScalarExpressionNodeSqrt::evaluateImplementation() const
{
  using std::sqrt;
  const auto lhs = _lhs->toScalar();
  SM_ASSERT_GT(std::runtime_error, lhs, 0.0, "");
  return sqrt(lhs);
}

void ScalarExpressionNodeSqrt::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::sqrt;
  const auto R = 1./(2.*sqrt(_lhs->toScalar()));
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeLog::evaluateImplementation() const
{
  using std::log;
  const auto lhs = _lhs->toScalar();
  SM_ASSERT_GT(std::runtime_error, lhs, 0.0, "");
  return log(lhs);
}

void ScalarExpressionNodeLog::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  const auto R = 1./(_lhs->toScalar());
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeExp::evaluateImplementation() const
{
  using std::exp;
  return exp(_lhs->toScalar());
}

void ScalarExpressionNodeExp::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::exp;
  const auto R = exp(_lhs->toScalar());
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeAtan::evaluateImplementation() const
{
  using std::atan;
  return atan(_lhs->toScalar());
}

void ScalarExpressionNodeAtan::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  const auto lhss = _lhs->toScalar();
  const auto R = 1./(1. + lhss * lhss);
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeAtan2::evaluateImplementation() const
{
  using std::atan2;
  return atan2(_lhs->toScalar(), _rhs->toScalar());
}

void ScalarExpressionNodeAtan2::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  // _rhs corresponds to x, _lhs to y
  const auto lhs = _lhs->toScalar();
  const auto rhs = _rhs->toScalar();
  const double factor = 1./(lhs*lhs + rhs*rhs);
  const auto R = -lhs*factor;
  const auto L = rhs*factor;
  _lhs->evaluateJacobians(outJacobians.apply(L));
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeAcos::evaluateImplementation() const
{
  using std::acos;
  auto lhss = _lhs->toScalar();
  SM_ASSERT_GE_LE(Exception, lhss, -1.0, 1.0, "");
  return acos(lhss);
}

void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::sqrt;
  const auto lhss = _lhs->toScalar();
  SM_ASSERT_LE(std::runtime_error, lhss, 1.0, "");
  const auto R = -1./sqrt(1. - lhss*lhss);
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeAcosSquared::evaluateImplementation() const
{
  using std::acos;
  const auto lhss = _lhs->toScalar();
  SM_ASSERT_LE(Exception, lhss, 1.0, "");
  SM_ASSERT_GE(Exception, lhss, -1.0, "");
  auto tmp = acos(lhss);
  return tmp*tmp;
}

void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::sqrt;
  using std::acos;

  const auto lhss = _lhs->toScalar();
  auto pow1 = lhss - 1.0;
  auto pow2 = pow1 * pow1;

  const double R = pow2 < std::numeric_limits<double>::epsilon() ?
      -2.0 + 2.0/3.0*pow1 : // series expansion at x = 1
      -2.0*acos(lhss)/sqrt(1.0 - lhss*lhss);
  SM_ASSERT_FALSE(Exception, std::isnan(R), "");

  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodeInverseSigmoid::evaluateImplementation() const
{
  using std::exp;
  const auto lhss = _lhs->toScalar();
  return _height / (exp((lhss - _shift) * _scale) + 1.0);
}

void ScalarExpressionNodeInverseSigmoid::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::exp;

  const auto lhss = _lhs->toScalar();
  const double threshold = 50;

  double R;
  if (lhss > threshold) {
    // approximate with Taylor expansion since exponents become too big otherwise
    auto den = exp(_scale*_shift) + exp(_scale*threshold);
    auto denSq = den*den;
    R = - _height*_scale*exp(_scale*(threshold+_shift)) / denSq +
        _height*_scale*_scale*(lhss-threshold)*exp(_scale*(_shift+threshold))
        *(exp(threshold*_scale) - exp(_scale*_shift)) / (denSq*den);
  }
  else {
    auto den = 1 + exp(_scale*(lhss-_shift));
    R = - _height*_scale*exp(_scale*(lhss-_shift)) / (den * den);
  }

  SM_ASSERT_FALSE(Exception, std::isnan(R), "");
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionNodePower::evaluateImplementation() const
{
  using std::pow;
  const auto lhss = _lhs->toScalar();
  return pow(lhss, _power);
}

void ScalarExpressionNodePower::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::pow;
  const auto lhss = _lhs->toScalar();
  const auto R = _power * pow(lhss, _power-1);
  SM_ASSERT_FALSE_DBG(Exception, std::isnan(R), "");
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

double ScalarExpressionPiecewiseExpression::evaluateImplementation() const
{
  if (_useFirst()) {
    return _e1->toScalar();
  } else {
    return _e2->toScalar();
  }
}

void ScalarExpressionPiecewiseExpression::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  if (_useFirst()) {
    return _e1->evaluateJacobians(outJacobians);
  } else {
    return _e2->evaluateJacobians(outJacobians);
  }
}

} /* namespace aslam */
} /* namespace backend */


#endif /* INCLUDE_ASLAM_BACKEND_IMPLEMENTATION_SCALAREXPRESSIONNODE_HPP_ */
