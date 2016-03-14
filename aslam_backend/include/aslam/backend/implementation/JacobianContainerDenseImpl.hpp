#include <aslam/backend/JacobianContainerDense.hpp>
#include <sm/assert_macros.hpp>

namespace aslam {
namespace backend {

template <typename Container>
JacobianContainerDense<Container>::JacobianContainerDense(int rows, int cols)
    : JacobianContainer(rows), _jacobian(rows, cols)
{
  clear();
}

/// \brief Apply the chain rule to the set of Jacobians.
/// This may change the number of rows of this set of Jacobians
/// by multiplying through by df_dx on the left.
template <typename Container>
void JacobianContainerDense<Container>::applyChainRule(const Eigen::MatrixXd& df_dx)
{
  SM_ASSERT_EQ(Exception, df_dx.cols(), _jacobian.rows(), "Invalid matrix product");
  _jacobian = df_dx * _jacobian;
}

template <typename Container>
bool JacobianContainerDense<Container>::isFinite(const DesignVariable& dv) const
{
  SM_ASSERT_GE_LE(Exception, dv.columnBase(), 0, _jacobian.cols() - dv.minimalDimensions(), "");
  return _jacobian.block(0, dv.columnBase(), _jacobian.rows(), dv.minimalDimensions()).allFinite();
}

template <typename Container>
Eigen::MatrixXd JacobianContainerDense<Container>::Jacobian(const DesignVariable* dv) const
{
  SM_ASSERT_GE_LE(Exception, dv->columnBase(), 0, _jacobian.cols() - dv->minimalDimensions(), "");
  return _jacobian.block(0, dv->columnBase(), _jacobian.rows(), dv->minimalDimensions());
}

/// \brief Clear the contents of this container
template <typename Container>
void JacobianContainerDense<Container>::clear()
{
  _jacobian.setZero();
}


template <typename Container>
inline void JacobianContainerDense<Container>::add(DesignVariable* dv, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian)
{
  if (!dv->isActive())
    return;
  SM_ASSERT_GE_LE(Exception, dv->columnBase(), 0, _jacobian.cols() - Jacobian.cols(), "");
  SM_ASSERT_EQ(Exception, _jacobian.rows(), Jacobian.rows(), "");
  _jacobian.block( 0, dv->columnBase(), Jacobian.rows(), Jacobian.cols() ) += _scale*Jacobian;
}

template <typename Container>
template<typename DERIVED>
void JacobianContainerDense<Container>::add(const JacobianContainerDense& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule /*= nullptr*/)
{

}


// Explicit template instantiation
extern template void JacobianContainerDense<Eigen::MatrixXd&>::add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian);
extern template bool JacobianContainerDense<Eigen::MatrixXd&>::isFinite(const DesignVariable& dv) const;
extern template void JacobianContainerDense<Eigen::MatrixXd&>::applyChainRule(const Eigen::MatrixXd& df_dx);
extern template void JacobianContainerDense<Eigen::MatrixXd&>::clear();

} // namespace backend
} // namespace aslam
