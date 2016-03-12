#ifndef ASLAM_JACOBIAN_CONTAINER_HPP
#define ASLAM_JACOBIAN_CONTAINER_HPP

#include <sparse_block_matrix/sparse_block_matrix.h>
#include <aslam/Exceptions.hpp>
#include <map>
#include <set>
#include "DesignVariable.hpp"
#include "backend.hpp"
#include "util/CommonDefinitions.hpp"

namespace aslam {
  namespace backend {

    class JacobianContainer {
    public:

      JacobianContainer(int rows) : _rows(rows) { }
      virtual ~JacobianContainer() { }

      /// \brief Add a jacobian to the list. If the design variable is not active,
      /// discard the value.
      virtual void add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian) = 0;

      /// \brief Apply the chain rule to the set of Jacobians.
      /// This may change the number of rows of this set of Jacobians
      /// by multiplying through by df_dx on the left.
      virtual void applyChainRule(const Eigen::MatrixXd& df_dx) = 0;

      /// \brief Gets a dense matrix with the Jacobians. The Jacobian ordering matches the sort order.
      virtual Eigen::MatrixXd asDenseMatrix() const = 0;

      /// \brief Clear the contents of this container
      virtual void clear() = 0;

      /// \brief Sets the scale
      void setScale(const double scale) { _scale = scale; }

      /// Check whether the entries corresponding to design variable \p dv are finite
      virtual bool isFinite(const DesignVariable& dv) const = 0;

      /// \brief How many rows does this set of Jacobians have?
      int rows() const { return _rows; }

    protected:

      /// \brief The number of rows for this set of Jacobians
      int _rows;

      /// \brief A scalar factor applied to the \p add() method
      double _scale = 1.0;

    };

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_JACOBIAN_CONTAINER_HPP */
