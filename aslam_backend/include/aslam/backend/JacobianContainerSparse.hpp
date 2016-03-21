#ifndef ASLAM_JACOBIAN_CONTAINER_SPARSE_HPP
#define ASLAM_JACOBIAN_CONTAINER_SPARSE_HPP

#include <sparse_block_matrix/sparse_block_matrix.h>
#include <aslam/Exceptions.hpp>
#include <map>
#include <set>
#include "DesignVariable.hpp"
#include "JacobianContainer.hpp"
#include "backend.hpp"
#include "util/CommonDefinitions.hpp"

namespace aslam {
  namespace backend {

    class JacobianContainerSparse : public JacobianContainer {
    public:
      SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

      /// \brief The map type for storing Jacobians. Sorting the list by block index
      ///        simplifies computing the upper-diagonal of the Hessian matrix.
      typedef std::map<DesignVariable*, Eigen::MatrixXd, DesignVariable::BlockIndexOrdering> map_t;
      typedef DesignVariable::set_t set_t;

      using JacobianContainer::JacobianContainer;
      virtual ~JacobianContainerSparse();

      /**
       * \brief Evaluate the Hessian and the RHS of Gauss-Newton given the error and inverse covariance.
       *
       * Only the upper triangular portion of the sparse matrix will be filled in.
       * This is the full Hessian matrix of the whole optimization problem. The correct index
       * for each design variable can be found using dv.blockIndex
       *
       * @param e          The error used for evaluation of the Hessian
       * @param invR       The inverse uncertainty used for evaluation of the Hessian
       * @param outHessian After evaluation this Hessian matrix will be filled in with J^T inv(R) J
       * @param outRhs     After evaluation this vector will be filled in with J^T inv(R) e
       */
      void evaluateHessian(const Eigen::VectorXd& e, const Eigen::MatrixXd& invR, SparseBlockMatrix& outHessian, Eigen::VectorXd& outRhs) const;

      /// \brief Add the rhs container to this one.
      template<typename DERIVED = Eigen::MatrixXd>
      void add(const JacobianContainerSparse& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule = nullptr);

      /// \brief Add the rhs container to this one. Alternative approach suitable for large left-hand sides
      template<typename DERIVED = Eigen::MatrixXd>
      void addLargeLhs(const JacobianContainerSparse& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule = nullptr);

      /// \brief Add a jacobian to the list. If the design variable is not active,
      /// discard the value.
      virtual void add(DesignVariable* designVariable, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian) override;

      /// \brief how many design variables does this jacobian container represent.
      size_t numDesignVariables() const;

      /// \brief Get design variable i.
      DesignVariable* designVariable(size_t i);

      /// \brief Get design variable i.
      const DesignVariable* designVariable(size_t i) const;

      map_t::const_iterator begin() const;
      map_t::const_iterator end() const;

      map_t::iterator begin();
      map_t::iterator end();


      /// Check whether the entries corresponding to design variable \p dv are finite
      virtual bool isFinite(const DesignVariable& dv) const override;

      /// Get the Jacobian associated with a particular design variable \p dv
      const Eigen::MatrixXd& Jacobian(const DesignVariable* dv) const;

      /// \brief Apply the chain rule to the set of Jacobians.
      /// This may change the number of rows of this set of Jacobians
      /// by multiplying through by df_dx on the left.
      void applyChainRule(const Eigen::MatrixXd& df_dx);

      /// \brief Clear the contents of this container
      void clear();

      /// \brief Clean and set the number of rows
      void reset(int rows);
      
      /// \brief Gets a sparse matrix with the Jacobians. The matrix is, in fact, dense
      ///        and the Jacobian ordering matches the sort order.
      SparseBlockMatrix asSparseMatrix() const;

      /// \brief Get a sparse matrix with the Jacobians. This uses the column block indices
      ///        to build the sparse, full width jacobian matrix
      SparseBlockMatrix asSparseMatrix(const std::vector<int>& colBlockIndices) const;

      /// \brief Gets a dense matrix with the Jacobians. The Jacobian ordering matches the sort order.
      virtual Eigen::MatrixXd asDenseMatrix() const override;

      /// \brief Gets a dense matrix with the Jacobians. This uses the column block indices
      ///        to build the sparse, full width jacobian matrix
      Eigen::MatrixXd asDenseMatrix(const std::vector<int>& colBlockIndices) const;

      /// The number of columns in the compressed Jacobian. Warning: this is expensive.
      int cols() const;
    private:

      void buildCorrelatedHessianBlock(const Eigen::VectorXd& e,
                                       const Eigen::MatrixXd& J1, int j1_block,
                                       SparseBlockMatrix& outHessian,
                                       map_t::const_iterator it, map_t::const_iterator it_end) const;

      void buildHessianBlock(const Eigen::VectorXd& e,
                             SparseBlockMatrix& outHessian,
                             Eigen::VectorXd& outRhs,
                             map_t::const_iterator it, map_t::const_iterator it_end) const;

      /// \brief The list of design variables.
      map_t _jacobianMap;
    };


    inline void JacobianContainerSparse::add(DesignVariable* dv, const Eigen::Ref<const Eigen::MatrixXd>& Jacobian)
    {
//
//      static Eigen::MatrixXd var;
//      var = Jacobian;
      sm::timing::DummyTimer timer("JacobianContainer::add", false);
      SM_ASSERT_EQ(Exception, Jacobian.rows(), _rows, "The Jacobian must have the same number of rows as this container");
      SM_ASSERT_EQ(Exception, Jacobian.cols(), dv->minimalDimensions(), "The Jacobian must have the same number of cols as dv->minimalDimensions()");
      // If the designe variable isn't active. Don't bother adding it.
      if (! dv->isActive())
        return;
      SM_ASSERT_GE_DBG(Exception, dv->blockIndex(), 0, "The design variable is active but the block index is less than zero.");
      map_t::iterator it = _jacobianMap.find(dv);
      if (it == _jacobianMap.end()) {
        _jacobianMap.insert(_jacobianMap.end(), std::make_pair(dv, Eigen::MatrixXd(Jacobian.template cast<double>())));
      } else {
        SM_ASSERT_TRUE_DBG(Exception, it->first == dv, "Two design variables had the same block index but different pointer values");
        it->second += Jacobian.template cast<double>();
      }
    }

    template<typename DERIVED>
    void JacobianContainerSparse::add(const JacobianContainerSparse& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule /*= nullptr*/)
    {
      SM_ASSERT_EQ(Exception, _rows, rhs._rows, "The JacobianContainers cannot be added. They don't have the same number of rows.");
      if (applyChainRule != nullptr)
        SM_ASSERT_EQ(Exception, applyChainRule->cols(), rhs._rows, "Wrong dimension of chain rule matrix");
      // Merge the two maps.
      // They are sorted by block it->first->blockIndex() so we can be smart about this.
      map_t::iterator lt = _jacobianMap.begin();
      const map_t::iterator lt_end = _jacobianMap.end();
      map_t::const_iterator rt = rhs._jacobianMap.begin();
      map_t::const_iterator rt_end = rhs._jacobianMap.end();
      bool done = rt == rt_end;
      for (; lt != lt_end && !done; ++lt) {
        // The maps are sorted by block index. FFWD the rhs map
        // inserting elements until we find the next possible
        // equal element.
        while (!done && rt->first->blockIndex() < lt->first->blockIndex()) {
          _jacobianMap.insert(lt, std::make_pair(rt->first, applyChainRule == nullptr ? rt->second : (*applyChainRule)*rt->second));
          ++rt;
          done = (rt == rt_end);
        }
        // If these keys match the Jacobians add
        if (!done && rt->first->blockIndex() == lt->first->blockIndex()) {
          SM_ASSERT_TRUE_DBG(Exception, rt->first == lt->first, "Two design variables had the same block index but different pointer values");
          // add the Jacobians.
          lt->second += applyChainRule == nullptr ? rt->second : (*applyChainRule)*rt->second;
        }
      }
      // Now the lhs list is done...add the remaining elements of the rhs list.
      for (; rt != rt_end; rt++) {
        map_t::iterator it = _jacobianMap.insert(lt, std::make_pair(rt->first, applyChainRule == nullptr ? rt->second : (*applyChainRule)*rt->second));
      }
    }

    template<typename DERIVED>
    void JacobianContainerSparse::addLargeLhs(const JacobianContainerSparse& rhs, const Eigen::MatrixBase<DERIVED>* applyChainRule /*= nullptr*/)
    {
      for (auto dvJacPair : rhs)
        add(dvJacPair.first, applyChainRule == nullptr ? dvJacPair.second : (*applyChainRule)*dvJacPair.second);
    }


  } // namespace backend
} // namespace aslam


#endif /* ASLAM_JACOBIAN_CONTAINER_SPARSE_HPP */
