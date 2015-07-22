#ifndef ASLAM_BACKEND_GENERIC_MATRIX_EXPRESSION_HPP
#define ASLAM_BACKEND_GENERIC_MATRIX_EXPRESSION_HPP

#include <set>
#include <type_traits>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <aslam/backend/JacobianContainer.hpp>
#include "ScalarExpression.hpp"
#include "ScalarExpressionNode.hpp"
#include "GenericMatrixExpressionNode.hpp"
#include "OperationResultNodes.hpp"
#include "Differential.hpp"

namespace aslam {
namespace backend {

namespace internal {
template<typename TNode>
struct GenericMatrixNodeTraits {
  typedef TNode node_t;
  typedef typename boost::shared_ptr<node_t> node_ptr_t;
  typedef typename node_t::matrix_t matrix_t;
  typedef typename node_t::value_t value_t;
  typedef typename node_t::tangent_vector_t tangent_vector_t;
  typedef typename node_t::differential_t differential_t;
  typedef typename node_t::constant_t constant_t;
};
}

template<int IRows, int ICols, typename TScalar = double, typename TNode = GenericMatrixExpressionNode<IRows, ICols, TScalar> >
class GenericMatrixExpression {
 public:
  typedef GenericMatrixExpression self_t;
  typedef GenericMatrixExpressionNode<IRows, ICols, TScalar> default_node_t;
  typedef TNode node_t;
  typedef GenericMatrixExpression<IRows, ICols, TScalar, default_node_t> default_self_t;

  typedef TScalar scalar_t;
  typedef internal::GenericMatrixNodeTraits<node_t> node_traits_t;
  typedef typename node_traits_t::node_ptr_t node_ptr_t;
  typedef typename node_traits_t::matrix_t matrix_t;
  typedef typename node_traits_t::value_t value_t;
  typedef typename node_traits_t::tangent_vector_t tangent_vector_t;
  typedef typename node_traits_t::differential_t differential_t;
  typedef typename node_traits_t::constant_t constant_t;

  /// \brief initialize from an existing node.
  GenericMatrixExpression(node_ptr_t root);

  /// \brief Initialize from an existing node. The node will not be deleted.
  GenericMatrixExpression(node_t * root);

  /// \brief Initialize a constant expression from an matrix. The matrix will be copied.
  template<typename DERIVED>
  GenericMatrixExpression(const Eigen::MatrixBase<DERIVED> & mat);

  virtual ~GenericMatrixExpression() {
  }

  /// \brief Evaluate the full Eigen matrix.
  matrix_t toFullMatrix() const;

  /// \brief Evaluate the full Eigen matrix.
  matrix_t evaluate() const {
    return toFullMatrix();
  }

  /// \brief return the expression representing the Moore-Penrose-Inverse of this expression.
  GenericMatrixExpression<ICols, IRows, TScalar> inverse() const;

  /// \brief return the transposed matrix expression.
  GenericMatrixExpression<ICols, IRows, TScalar> transpose() const;

  template<int IColsOther, typename TOtherNode>
  GenericMatrixExpression<IRows, IColsOther, TScalar> operator*(const GenericMatrixExpression<ICols, IColsOther, TScalar, TOtherNode> & other) const;

  inline operator GenericMatrixExpression<IRows, ICols, TScalar>() const;

  default_self_t operator*(const ScalarExpression & scalarExpression) const;
  default_self_t operator*(TScalar scalar) const;
  default_self_t operator-() const;
  template<typename TOtherNode>
  default_self_t operator+(const GenericMatrixExpression<IRows, ICols, TScalar, TOtherNode> & other) const;
  template<typename TOtherNode>
  default_self_t operator-(const GenericMatrixExpression<IRows, ICols, TScalar, TOtherNode> & other) const;

  template<int IColsOther, typename TOtherNode, typename std::enable_if<IRows == 3 && (ICols == 1 || IColsOther == 1), int>::type = 0>
  GenericMatrixExpression<3, (ICols == 1 ? IColsOther : ICols), TScalar> cross(const GenericMatrixExpression<3, IColsOther, TScalar, TOtherNode> & other) const;

  /// \brief Evaluate the Jacobians
  void evaluateJacobians(JacobianContainer & outJacobians) const;
  void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
  void evaluateJacobians(JacobianContainer & outJacobians, const differential_t & diff) const;

  void getDesignVariables(DesignVariable::set_t & designVariables) const;

  ScalarExpression toScalarExpression() const;
  template <int RowIndex, int ColIndex>
  ScalarExpression toScalarExpression() const;

  inline node_ptr_t root() const {
    return _root;
  }

  inline static self_t constant(int rows = IRows, int cols = ICols) {
    return GenericMatrixExpression(new constant_t(rows, cols));
  }
 protected:
  GenericMatrixExpression() {
  }
 private:
  node_ptr_t _root;

 public:
  template<typename TDerived, typename TOperand>
  class UnaryOperationResult : public UnaryOperationResultNode<TDerived, TOperand, self_t, typename node_t::differential_t::domain_t, scalar_t> {
  };

  template<typename TDerived, typename TLhs, typename TRhs>
  class BinaryOperationResult : public BinaryOperationResultNode<TDerived, TLhs, TRhs, self_t, typename node_t::differential_t::domain_t, scalar_t> {
  };
};

}  // namespace backend
}  // namespace aslam

#include "implementation/GenericMatrixExpression.hpp"

#endif /* ASLAM_BACKEND_MATRIX_EXPRESSION_HPP */
