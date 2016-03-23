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
template<typename DERIVED>
inline void JacobianContainerDense<Container>::addImpl(DesignVariable* dv, const Eigen::MatrixBase<DERIVED>& Jacobian, const bool isIdentity)
{
  if (!dv->isActive())
    return;

  SM_ASSERT_GE_LE(Exception, dv->columnBase(), 0, _jacobian.cols() - Jacobian.cols(), "Check that column base of design variable is set correctly");

  if (this->chainRuleEmpty()) // stack empty
  {
    SM_ASSERT_EQ(Exception, _jacobian.rows(), Jacobian.rows(), "");
    _jacobian.block( 0, dv->columnBase(), Jacobian.rows(), Jacobian.cols() ) += Jacobian.template cast<double>();
  }
  else
  {
    auto CR = this->chainRuleMatrix();
    SM_ASSERT_EQ_DBG(Exception, _jacobian.rows(), CR.rows(), ""); // This is very unlikely since the chain rule size is checked in JacobianContainer::apply()

    if (!isIdentity) {// TODO: cleanup
      SM_ASSERT_EQ_DBG(Exception, CR.cols(), Jacobian.rows(), "Invalid matrix product of chain rule matrix and Jacobian");
      _jacobian.block( 0, dv->columnBase(), CR.rows(), Jacobian.cols() ).noalias() += CR*Jacobian.template cast<double>();
    }else
      _jacobian.block( 0, dv->columnBase(), CR.rows(), CR.cols() ) += CR;
  }
}

template <typename Container>
template<typename DERIVED>
void JacobianContainerDense<Container>::add(const JacobianContainerDense& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule /*= nullptr*/)
{
  SM_THROW(NotImplementedException, __PRETTY_FUNCTION__ << " not implemented");
}

template <typename Container>
void JacobianContainerDense<Container>::add(DesignVariable* dv, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian)
{
  this->addImpl(dv, Jacobian, false);
}

template <typename Container>
void JacobianContainerDense<Container>::add(DesignVariable* designVariable)
{
  this->addImpl(designVariable, Eigen::MatrixXd::Identity(this->rows(), designVariable->minimalDimensions()), true);
}


// Explicit template instantiation
extern template void JacobianContainerDense<Eigen::MatrixXd&>::add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian);
extern template void JacobianContainerDense<Eigen::MatrixXd&>::add(DesignVariable* designVariable);
extern template bool JacobianContainerDense<Eigen::MatrixXd&>::isFinite(const DesignVariable& dv) const;
extern template void JacobianContainerDense<Eigen::MatrixXd&>::clear();

} // namespace backend
} // namespace aslam
