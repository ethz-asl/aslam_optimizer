#include <vector>

#include <vector>

#include <aslam/backend/JacobianContainerDense.hpp>
#include <sm/assert_macros.hpp>

namespace aslam {
namespace backend {

// Explicit template instantiation
template void JacobianContainerDense<Eigen::MatrixXd&>::add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian);
template bool JacobianContainerDense<Eigen::MatrixXd&>::isFinite(const DesignVariable& dv) const;
template void JacobianContainerDense<Eigen::MatrixXd&>::clear();

} // namespace backend
} // namespace aslam

