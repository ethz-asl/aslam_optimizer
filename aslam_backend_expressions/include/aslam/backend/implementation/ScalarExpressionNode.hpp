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

/// \brief Evaluate the Jacobians and apply the chain rule.
void ScalarExpressionNode::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  SM_ASSERT_EQ_DBG(Exception, applyChainRule.cols(), 1, "The chain rule matrix must have one columns");
  evaluateJacobiansImplementation(outJacobians, applyChainRule);
}



double ScalarExpressionNodeMultiply::evaluateImplementation() const
{
  return _lhs->toScalar() * _rhs->toScalar();
}

void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  Eigen::Matrix<double, 1, 1> L(1,1), R(1,1);
  L(0,0) = _lhs->toScalar();
  R(0,0) = _rhs->toScalar();

  _lhs->evaluateJacobians(outJacobians.apply(R));
  _rhs->evaluateJacobians(outJacobians.apply(L));
}

void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}

double ScalarExpressionNodeDivide::evaluateImplementation() const
{
  return _lhs->toScalar() / _rhs->toScalar();
}

void ScalarExpressionNodeDivide::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  Eigen::Matrix<double, 1, 1> L(1,1), R(1,1);
  const auto rhs_rec = 1./_rhs->toScalar();
  L(0,0) = rhs_rec;
  R(0,0) = -_lhs->toScalar() * rhs_rec * rhs_rec;
  _lhs->evaluateJacobians(outJacobians.apply(L));
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeDivide::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}



double ScalarExpressionNodeNegated::evaluateImplementation() const
{
  return -_rhs->toScalar();
}

void ScalarExpressionNodeNegated::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  Eigen::Matrix<double, 1, 1> R(1,1);
  R(0,0) = -1;
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeNegated::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


double ScalarExpressionNodeAdd::evaluateImplementation() const
{
  return _lhs->toScalar() + _multiplyRhs * _rhs->toScalar();
}

void ScalarExpressionNodeAdd::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  Eigen::Matrix<double, 1, 1> R(1,1);
  R(0,0) = _multiplyRhs;
  _lhs->evaluateJacobians(outJacobians);
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeAdd::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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
  Eigen::Matrix<double, 1, 1> R(1,1);
  R(0,0) = 1./(2.*sqrt(_lhs->toScalar()));
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeSqrt::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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
  Eigen::Matrix<double, 1, 1> R(1,1);
  R(0,0) = 1./(_lhs->toScalar());
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeLog::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


double ScalarExpressionNodeExp::evaluateImplementation() const
{
  using std::exp;
  return exp(_lhs->toScalar());
}

void ScalarExpressionNodeExp::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::exp;
  Eigen::Matrix<double, 1, 1> R(1,1);
  R(0,0) = exp(_lhs->toScalar());
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeExp::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


double ScalarExpressionNodeAtan::evaluateImplementation() const
{
  using std::atan;
  return atan(_lhs->toScalar());
}

void ScalarExpressionNodeAtan::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  Eigen::Matrix<double, 1, 1> R(1,1);
  const auto lhss = _lhs->toScalar();
  R(0,0) = 1./(1. + lhss * lhss);
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeAtan::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


double ScalarExpressionNodeAtan2::evaluateImplementation() const
{
  using std::atan2;
  return atan2(_lhs->toScalar(), _rhs->toScalar());
}

void ScalarExpressionNodeAtan2::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  // _rhs corresponds to x, _lhs to y
  Eigen::Matrix<double, 1, 1> R(1,1);
  Eigen::Matrix<double, 1, 1> L(1,1);
  const auto lhs = _lhs->toScalar();
  const auto rhs = _rhs->toScalar();
  const double factor = 1./(lhs*lhs + rhs*rhs);
  R(0,0) = -lhs*factor;
  L(0,0) = rhs*factor;
  _lhs->evaluateJacobians(outJacobians.apply(L));
  _rhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeAtan2::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


double ScalarExpressionNodeAcos::evaluateImplementation() const
{
  using std::acos;
  auto lhss = _lhs->toScalar();
  SM_ASSERT_LE(Exception, lhss, 1.0, "");
  SM_ASSERT_GE(Exception, lhss, -1.0, "");
  return acos(lhss);
}

void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
{
  using std::sqrt;
  Eigen::Matrix<double, 1, 1> R(1,1);
  const auto lhss = _lhs->toScalar();
  SM_ASSERT_LE(std::runtime_error, lhss, 1.0, "");
  R(0,0) = -1./sqrt(1. - lhss*lhss);
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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

  Eigen::Matrix<double, 1, 1> R(1,1);
  const auto lhss = _lhs->toScalar();
  auto pow1 = lhss - 1.0;
  auto pow2 = pow1 * pow1;

  if (pow2 < std::numeric_limits<double>::epsilon())   // series expansion at x = 1
  {
    R(0,0) = -2.0 + 2.0/3.0*pow1;
  }
  else
  {
    R(0,0) = -2.0*acos(lhss)/sqrt(1.0 - lhss*lhss);
  }
  SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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

  Eigen::Matrix<double, 1, 1> R(1,1);
  const auto lhss = _lhs->toScalar();
  const double threshold = 50;

  if (lhss > threshold) {
    // approximate with Taylor expansion since exponents become too big otherwise
    auto den = exp(_scale*_shift) + exp(_scale*threshold);
    auto denSq = den*den;
    R(0,0) = - _height*_scale*exp(_scale*(threshold+_shift)) / denSq +
        _height*_scale*_scale*(lhss-threshold)*exp(_scale*(_shift+threshold))
        *(exp(threshold*_scale) - exp(_scale*_shift)) / (denSq*den);
  }
  else {
    auto den = 1 + exp(_scale*(lhss-_shift));
    R(0,0) = - _height*_scale*exp(_scale*(lhss-_shift)) / (den * den);
  }

  SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodeInverseSigmoid::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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
  Eigen::Matrix<double, 1, 1> R(1,1);
  const auto lhss = _lhs->toScalar();
  R(0,0) = _power * pow(lhss, _power-1);
  SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
  _lhs->evaluateJacobians(outJacobians.apply(R));
}

void ScalarExpressionNodePower::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
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

void ScalarExpressionPiecewiseExpression::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
{
  evaluateJacobiansImplementation(outJacobians.apply(applyChainRule));
}


} /* namespace aslam */
} /* namespace backend */


#endif /* INCLUDE_ASLAM_BACKEND_IMPLEMENTATION_SCALAREXPRESSIONNODE_HPP_ */
