#ifndef ASLAM_BACKEND_GENERIC_MATRIX_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_GENERIC_MATRIX_EXPRESSION_NODE_HPP
#include <aslam/backend/JacobianContainer.hpp>
#include <aslam/backend/Differential.hpp>

namespace aslam {
namespace backend {

template<int IRows, int ICols, typename TScalar>
class GenericMatrixExpressionNode {
 public:
  typedef GenericMatrixExpressionNode<IRows, ICols, TScalar> self_t;
  typedef Eigen::Matrix<TScalar, IRows, ICols> matrix_t;
  typedef matrix_t value_t;
  typedef Eigen::Matrix<TScalar, IRows, ICols> tangent_vector_t;
  typedef Differential<tangent_vector_t, TScalar> differential_t;
  typedef boost::shared_ptr<self_t> ptr_t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  mutable matrix_t _currentValue;
  mutable bool _valueDirty;

  inline GenericMatrixExpressionNode(const matrix_t & value)
      : _currentValue(value),
        _valueDirty(false) {
  }
 public:
  GenericMatrixExpressionNode(int rows = IRows, int cols = ICols, bool valueDirty = true)
      : _currentValue(rows, cols),
        _valueDirty(valueDirty) {
  }
  virtual ~GenericMatrixExpressionNode() {
  }
  ;

  inline matrix_t & getCurrentValue() {
    return _currentValue;
  }
  inline const matrix_t & getCurrentValue() const {
    return _currentValue;
  }
  inline const matrix_t & evaluate() const {
    evaluateImplementation();
    /* TODO activate value caching again.
    if (!isConstant() && _valueDirty) {
      evaluateImplementation();
      _valueDirty = false;
    };*/
    return _currentValue;
  }

  void evaluateJacobians(JacobianContainer & outJacobians, const differential_t & diff) const {
    evaluateJacobiansImplementation(outJacobians, diff);
  }

  const matrix_t toMatrix() const {
    return evaluate();
  }
  ;

  void getDesignVariables(DesignVariable::set_t & designVariables) const {
    return getDesignVariablesImplementation(designVariables);
  }
  bool isConstant() const {
    return isConstantImplementation();
  }

  void inline invalidate() {
    _valueDirty = true;
  }
  ;
 protected:
  virtual void evaluateImplementation() const = 0;
  virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;
  virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const differential_t & diff) const = 0;
  virtual bool isConstantImplementation() const {
    return false;
  }

 public:
  class Constant : public GenericMatrixExpressionNode {
   public:
    template<typename DERIVED>
    Constant(const Eigen::MatrixBase<DERIVED> & value)
        : GenericMatrixExpressionNode(value) {
    }
    Constant(int rows = IRows, int cols = ICols)
        : GenericMatrixExpressionNode(rows, cols, false) {
    }
   protected:
    bool isConstantImplementation() const {
      return true;
    }
    virtual void evaluateImplementation() const {
    }
    ;
    virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const differential_t & diff) const {
    }
    virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const {
    }
  };
  typedef Constant constant_t;
};

}  // namespace backend
}  // namespace aslam

//#include "implementation/GenericMatrixExpressionNode.hpp"

#endif /* ASLAM_BACKEND_GENERIC_MATRIX_EXPRESSION_NODE_HPP */
