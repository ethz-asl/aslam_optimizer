#include <tuple>
#include <utility>

#include <aslam/backend/ScalarExpressionNode.hpp>

namespace aslam {
namespace backend {

template<int D>
void VectorExpressionNode<D>::evaluateJacobians(JacobianContainer & outJacobians) const {
  evaluateJacobiansImplementation(outJacobians);
}

template<int D>
void VectorExpressionNode<D>::evaluateJacobians(JacobianContainer & outJacobians, const differential_t & diff) const {
  evaluateJacobiansImplementationWithDifferential(outJacobians, diff);
}

template<int D>
void VectorExpressionNode<D>::evaluateJacobiansImplementationWithDifferential(JacobianContainer & outJacobians, const differential_t & chainRuleDifferentail) const {
  evaluateJacobiansImplementation(applyDifferentialToJacobianContainer(outJacobians, chainRuleDifferentail, getSize()));
}

template<int D>
void VectorExpressionNode<D>::getDesignVariables(DesignVariable::set_t & designVariables) const {
  getDesignVariablesImplementation(designVariables);
}

template<int D>
void ConstantVectorExpressionNode<D>::accept(ExpressionNodeVisitor& visitor) {
  visitor.visit("#", this);
}

template <int D>
void StackedScalarVectorExpressionNode<D>::accept(ExpressionNodeVisitor& visitor) {
  visitor.visit("#", this, std::tuple_cat(components));
}

template <int D>
typename StackedScalarVectorExpressionNode<D>::vector_t
StackedScalarVectorExpressionNode<D>::evaluateImplementation() const {
  vector_t res;
  int idx = 0;
  for (const auto& component : components) {
    res[idx++] = component->evaluate();
  }
  return res;
}

template <int D>
void StackedScalarVectorExpressionNode<D>::evaluateJacobiansImplementation(
    JacobianContainer& outJacobians) const {
  for (int i = 0; i < components.size(); ++i) {
    components.at(i)->evaluateJacobians(
        outJacobians, Eigen::Matrix<double, D, 1>::Unit(i));
  }
}

template <int D>
void StackedScalarVectorExpressionNode<D>::getDesignVariablesImplementation(
    DesignVariable::set_t& designVariables) const {
  for (const auto& component : components) {
    component->getDesignVariables(designVariables);
  }
}

template <int D>
void VectorExpressionNodeAddVector<D>::accept(ExpressionNodeVisitor& visitor) {
  visitor.visit("+", this, _lhs, _rhs);
}

template <int D>
typename VectorExpressionNodeAddVector<D>::vector_t
VectorExpressionNodeAddVector<D>::evaluateImplementation() const {
  return _lhs->evaluate() + _rhs->evaluate();
}

template <int D>
void VectorExpressionNodeAddVector<D>::evaluateJacobiansImplementation(
    JacobianContainer& outJacobians) const {
  _lhs->evaluateJacobians(outJacobians);
  _rhs->evaluateJacobians(outJacobians);
}

template <int D>
void VectorExpressionNodeAddVector<D>::getDesignVariablesImplementation(
    DesignVariable::set_t& designVariables) const {
  _lhs->getDesignVariables(designVariables);
  _rhs->getDesignVariables(designVariables);
}

template <int D>
void VectorExpressionNodeScalarMultiply<D>::getDesignVariablesImplementation(
    DesignVariable::set_t& designVariables) const {
  _p->getDesignVariables(designVariables);
  _s->getDesignVariables(designVariables);
}

template <int D>
typename VectorExpressionNodeScalarMultiply<D>::vector_t VectorExpressionNodeScalarMultiply<D>::evaluateImplementation()
    const {
  return _p->evaluate() * _s->evaluate();
}

template <int D>
void VectorExpressionNodeScalarMultiply<D>::evaluateJacobiansImplementation(
    JacobianContainer& outJacobians) const {
  _p->evaluateJacobians(outJacobians.apply(_s->evaluate()));
  _s->evaluateJacobians(outJacobians.apply(_p->evaluate()));
}

}  // namespace backend
}  // namespace aslam
