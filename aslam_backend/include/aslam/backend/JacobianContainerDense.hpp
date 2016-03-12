#ifndef ASLAM_JACOBIAN_CONTAINER_DENSE_HPP
#define ASLAM_JACOBIAN_CONTAINER_DENSE_HPP

#include <aslam/Exceptions.hpp>
#include <vector>
#include "DesignVariable.hpp"
#include "JacobianContainer.hpp"
#include "backend.hpp"
#include "util/CommonDefinitions.hpp"

namespace aslam {
  namespace backend {

    template <typename Container>
    class JacobianContainerDense : public JacobianContainer {
    public:
      SM_DEFINE_EXCEPTION(Exception, aslam::Exception);
      typedef Container container_t;

      JacobianContainerDense(int rows, int cols);
      JacobianContainerDense(Container jacobian) : JacobianContainer(jacobian.rows()), _jacobian(jacobian) { }
      virtual ~JacobianContainerDense() { }

      /// \brief Add the rhs container to this one.
      template<typename DERIVED = Eigen::MatrixXd>
      void add(const JacobianContainerDense& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule = nullptr);

      /// \brief Add a jacobian to the list. If the design variable is not active,
      /// discard the value.
      virtual void add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian) override;

      /// Check whether the entries corresponding to design variable \p dv are finite
      virtual bool isFinite(const DesignVariable& dv) const override;

      /// Get the Jacobian associated with a particular design variable \p dv
      Eigen::MatrixXd Jacobian(const DesignVariable* dv) const;

      /// \brief Apply the chain rule to the set of Jacobians.
      /// This may change the number of rows of this set of Jacobians
      /// by multiplying through by df_dx on the left.
      virtual void applyChainRule(const Eigen::MatrixXd& df_dx) override;

      /// \brief Clear the contents of this container
      virtual void clear() override;

      /// \brief Gets a dense matrix with the Jacobians. The Jacobian ordering matches the sort order.
      virtual Eigen::MatrixXd asDenseMatrix() const override { return _jacobian; }

    private:

      /// \brief The data
      Container _jacobian;
    };

  } // namespace backend
} // namespace aslam

#include "implementation/JacobianContainerDenseImpl.hpp"

#endif /* ASLAM_JACOBIAN_CONTAINER_DENSE_HPP */
