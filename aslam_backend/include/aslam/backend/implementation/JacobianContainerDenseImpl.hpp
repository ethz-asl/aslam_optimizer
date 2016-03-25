#include <type_traits>

#include <aslam/backend/JacobianContainerDense.hpp>
#include <sm/assert_macros.hpp>

#define JACOBIAN_CONTAINER_DENSE_TEMPLATE template <typename Container, int Rows>
#define JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE JacobianContainerDense<Container, Rows>

namespace aslam {
namespace backend {

JACOBIAN_CONTAINER_DENSE_TEMPLATE
template<typename dummy>
JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::JacobianContainerDense(int rows, int cols, dummy)
    : JacobianContainer(rows), _jacobian(rows, cols)
{
  SM_ASSERT_TRUE(Exception, Rows == Eigen::Dynamic || _jacobian.rows() == Rows, "");
  clear();
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
bool JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::isFinite(const DesignVariable& dv) const
{
  SM_ASSERT_GE_LE(Exception, dv.columnBase(), 0, _jacobian.cols() - dv.minimalDimensions(), "");
  return _jacobian.block(0, dv.columnBase(), _jacobian.rows(), dv.minimalDimensions()).allFinite();
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
Eigen::MatrixXd JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::Jacobian(const DesignVariable* dv) const
{
  SM_ASSERT_GE_LE(Exception, dv->columnBase(), 0, _jacobian.cols() - dv->minimalDimensions(), "");
  return _jacobian.block(0, dv->columnBase(), _jacobian.rows(), dv->minimalDimensions());
}

/// \brief Clear the contents of this container
JACOBIAN_CONTAINER_DENSE_TEMPLATE
void JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::clear()
{
  _jacobian.setZero();
}

/// \brief Hack to prevent Eigen assertion "YOU_ARE_TRYING_TO_USE_AN_INDEX_BASED_ACCESSOR_ON_AN_EXPRESSION_THAT_DOES_NOT_SUPPORT_THAT"
///        with fixed-row r1 and Identity functor r2.
///        This method is called for non-identity r2
template <typename L, typename R1, typename R2>
EIGEN_ALWAYS_INLINE void multiplyRhsAndAddToLhs(L&& l, R1& r1, R2& r2, std::false_type /*isIdentity*/)
{
  SM_ASSERT_EQ_DBG(Exception, r1.cols(), r2.rows(), "Invalid matrix product of chain rule matrix and Jacobian");
  l.noalias() += r1*r2;
}
/// \brief This method is called for identity r2
template <typename L, typename R1, typename R2>
EIGEN_ALWAYS_INLINE void multiplyRhsAndAddToLhs(L&& l, R1& r1, R2& /*r2*/, std::true_type /*isIdentity*/)
{
  l += r1.leftCols(l.cols());
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
template<bool IS_IDENTITY, typename MATRIX>
EIGEN_ALWAYS_INLINE void JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::addImpl(DesignVariable* dv, const MATRIX& Jacobian)
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
    const auto CR = this->chainRuleMatrix<Rows>();
    SM_ASSERT_EQ_DBG(Exception, _jacobian.rows(), CR.rows(), ""); // This is very unlikely since the chain rule size is checked in JacobianContainer::apply()
    multiplyRhsAndAddToLhs(_jacobian.block( 0, dv->columnBase(), CR.rows(), Jacobian.cols() ), CR, Jacobian.template cast<double>(), std::integral_constant<bool, IS_IDENTITY>());
  }
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
template<typename DERIVED>
void JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::add(const JacobianContainerDense& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule /*= nullptr*/)
{
  SM_THROW(NotImplementedException, __PRETTY_FUNCTION__ << " not implemented");
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
void JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::add(DesignVariable* dv, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian)
{
  this->addImpl<false>(dv, Jacobian);
}

JACOBIAN_CONTAINER_DENSE_TEMPLATE
void JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE::add(DesignVariable* designVariable)
{
  this->addImpl<true>(designVariable, Eigen::MatrixXd::Identity(this->rows(), designVariable->minimalDimensions()));
}


// Explicit template instantiation
extern template class JacobianContainerDense<Eigen::MatrixXd, Eigen::Dynamic>;
extern template class JacobianContainerDense<Eigen::MatrixXd, 1>;
extern template class JacobianContainerDense<Eigen::MatrixXd, 2>;
extern template class JacobianContainerDense<Eigen::MatrixXd&, Eigen::Dynamic>;
extern template class JacobianContainerDense<Eigen::MatrixXd&, 1>;
extern template class JacobianContainerDense<Eigen::MatrixXd&, 2>;

#undef JACOBIAN_CONTAINER_DENSE_TEMPLATE
#undef JACOBIAN_CONTAINER_DENSE_CLASS_TEMPLATE

} // namespace backend
} // namespace aslam
