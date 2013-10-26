/*
 * ExpressionErrorTerm.hpp
 *
 *  Created on: Jan 22, 2013
 *      Author: hannes
 */

#ifndef EXPRESSIONERRORTERM_HPP_
#define EXPRESSIONERRORTERM_HPP_

#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/GenericMatrixExpression.hpp>
#include <aslam/backend/GenericScalarExpression.hpp>

namespace aslam {
namespace backend {

namespace internal {
template<typename TExpression>
struct ExpressionDimensionTraits {
  enum {
    Dimension = TExpression::Dimension
  };
};

template<int IDimension>
struct ExpressionDimensionTraits<VectorExpression<IDimension> > {
  enum {
    Dimension = IDimension
  };
};
template<typename TScalar, int IDimension, typename TNode>
struct ExpressionDimensionTraits<GenericMatrixExpression<IDimension, 1, TScalar, TNode> > {
  enum {
    Dimension = IDimension
  };
};
template<typename TScalar>
struct ExpressionDimensionTraits<GenericScalarExpression<TScalar> > {
  enum {
    Dimension = 1
  };
};
}

template<typename TExpression, int IDimension = internal::ExpressionDimensionTraits<TExpression>::Dimension>
class ExpressionErrorTerm : public aslam::backend::ErrorTermFs<IDimension> {
 public:
  typedef ExpressionErrorTerm self_t;
  typedef aslam::backend::ErrorTermFs<IDimension> parent_t;
  typedef Eigen::Matrix<double, IDimension, 1> PointT;

  ExpressionErrorTerm(const TExpression & expression)
      : _expression(expression) {
    DesignVariable::set_t vSet;
    _expression.getDesignVariables(vSet);
    vector<DesignVariable *> vs;
    vs.reserve(vSet.size());
    std::copy(vSet.begin(), vSet.end(), back_inserter(vs));
    parent_t::setDesignVariables(vs);
  }
  virtual ~ExpressionErrorTerm() {
  }

  /// \brief evaluate the error term
  virtual double evaluateErrorImplementation() {
    auto error = _expression.evaluate();
    this->setError(error);
    auto tmp = (this->sqrtInvR() * error).eval();
    return tmp.dot(tmp);
  }

  /// \brief evaluate the jacobian
  virtual void evaluateJacobiansImplementation(JacobianContainer & jacobians) {
    _expression.evaluateJacobians(jacobians);
  }

  inline TExpression getExpression() {
    return _expression;
  }
 private:
  const TExpression _expression;
};

template<typename TExpression, int IDimension = internal::ExpressionDimensionTraits<TExpression>::Dimension>
inline ::boost::shared_ptr<ErrorTerm> toErrorTerm(TExpression expression) {
  return ::boost::shared_ptr<ErrorTerm>(new ExpressionErrorTerm<TExpression, IDimension>(expression));
}

template<typename TExpression, int IDimension = internal::ExpressionDimensionTraits<TExpression>::Dimension>
inline ::boost::shared_ptr<ErrorTerm> toErrorTerm(TExpression expression, const Eigen::Matrix<double, IDimension, IDimension> & RInv) {
  ::boost::shared_ptr<ErrorTerm> errorTerm = toErrorTerm(expression);
  errorTerm->vsSetInvR(RInv);
  return errorTerm;
}

}
}

#endif /* EXPRESSIONERRORTERM_HPP_ */
