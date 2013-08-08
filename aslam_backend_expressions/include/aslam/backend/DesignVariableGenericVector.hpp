#ifndef ASLAM_BACKEND_DESIGN_VARIABLE_GENERIC_VECTOR_HPP
#define ASLAM_BACKEND_DESIGN_VARIABLE_GENERIC_VECTOR_HPP

#include <aslam/backend/JacobianContainer.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include "GenericMatrixExpressionNode.hpp"

namespace aslam {
namespace backend {
template<int D>
class DesignVariableGenericVector : public DesignVariable, public GenericMatrixExpressionNode<D, 1, double> {
 public:
  typedef GenericMatrixExpressionNode<D, 1, double> base_t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<double, D, 1> vector_t;

  SM_DEFINE_EXCEPTION(Exception, std::runtime_error)
  ;

  DesignVariableGenericVector(vector_t v = vector_t::Zero())
      : base_t(v) {
  }
  virtual ~DesignVariableGenericVector() {
  }
  const vector_t & value() const {
    return base_t::evaluate();
  }
 protected:
  /// \brief Revert the last state update.
  virtual void revertUpdateImplementation() {
    this->_currentValue = _p_v;
  }
  /// \brief Update the design variable.
  virtual void updateImplementation(const double * dp, int size) {
    SM_ASSERT_EQ(std::runtime_error, size, D, "update size must match the vector dimension.")
    _p_v = this->_currentValue;
    this->_currentValue += Eigen::Map<const vector_t>(dp);
  }
  /// \brief what is the number of dimensions of the perturbation variable.
  virtual int minimalDimensionsImplementation() const {
    return D;
  }

  virtual void getParametersImplementation(Eigen::MatrixXd& value) const {
    value = this->evaluate();
  }

  virtual void setParametersImplementation(const Eigen::MatrixXd& value) {
    this->_currentValue = value;
    this->_valueDirty = false;
  }

  virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const {
    designVariables.insert((DesignVariable*) this);
  }
  ;
  virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const typename base_t::differential_t & diff) const {
    diff.addToJacobianContainer(outJacobians, (const DesignVariable *) this);
  }
  ;
  virtual void evaluateImplementation() const {
  }
 private:
  vector_t _p_v;
};

}  // namespace backend
}  // namespace aslam

#endif /* ASLAM_BACKEND_DESIGN_VARIABLE_GENERIC_VECTOR_HPP */
