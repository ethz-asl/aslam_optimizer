#ifndef ASLAM_JACOBIAN_CONTAINER_HPP
#define ASLAM_JACOBIAN_CONTAINER_HPP

#include <sparse_block_matrix/sparse_block_matrix.h>
#include <aslam/Exceptions.hpp>
#include <map>
#include <set>
#include "DesignVariable.hpp"
#include "backend.hpp"
#include "MatrixStack.hpp"
#include "util/CommonDefinitions.hpp"

namespace aslam {
  namespace backend {

    struct JacobianContainerChainRuleApplied {
      JacobianContainerChainRuleApplied(MatrixStack::PopGuard&& pg)
          : _guard(std::move(pg))
      {

      }
      inline operator JacobianContainer&();
     private:
      MatrixStack::PopGuard _guard;
    };

    class JacobianContainer : private MatrixStack {

     public:
      friend class JacobianContainerChainRuleApplied;

     public:

      JacobianContainer(int rows, const std::size_t maxNumMatrices = 100) : MatrixStack(maxNumMatrices, 9), _rows(rows) { }
      virtual ~JacobianContainer() { }

      /// \brief Add a jacobian to the list. If the design variable is not active, discard the value.
      virtual void add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian) = 0;

      /// \brief Add a jacobian to the list with identity chain rule. If the design variable is not active, discard the value.
      virtual void add(DesignVariable* designVariable) = 0;

      /// \brief Gets a dense matrix with the Jacobians. The Jacobian ordering matches the sort order.
      virtual Eigen::MatrixXd asDenseMatrix() const = 0;

      /// Check whether the entries corresponding to design variable \p dv are finite
      virtual bool isFinite(const DesignVariable& dv) const = 0;

      /// \brief How many rows does this set of Jacobians have?
      int rows() const { return _rows; }

      /// \brief Push a matrix \p mat to the top of the stack
      template <typename DERIVED>
      JacobianContainerChainRuleApplied apply(const Eigen::MatrixBase<DERIVED>& mat)
      {
        if (this->chainRuleEmpty())
          SM_ASSERT_EQ(Exception, this->rows(), mat.rows(), "");
        return JacobianContainerChainRuleApplied(this->pushWithGuard(mat));
      }

    protected:

      /// \brief Is the stack empty?
      bool chainRuleEmpty() const {
        return this->empty();
      }

      /// \brief Const getter for the chain rule matrix
      MatrixStack::ConstMap chainRuleMatrix() const {
        return this->top();
      }

      /// \brief Mutable getter for the chain rule matrix
      MatrixStack::Map chainRuleMatrix() {
        return this->top();
      }

    protected:
      /// \brief The number of rows for this set of Jacobians
      int _rows;

    };


    inline JacobianContainerChainRuleApplied::operator JacobianContainer&()
    {
      return static_cast<JacobianContainer&>(_guard.stack());
    }

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_JACOBIAN_CONTAINER_HPP */
