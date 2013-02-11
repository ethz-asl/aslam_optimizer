#include <aslam/backend/SparseQrLinearSystemSolver.hpp>

namespace aslam {
  namespace backend {
    SparseQrLinearSystemSolver::SparseQrLinearSystemSolver() :
        _factor(NULL) {
    }

    SparseQrLinearSystemSolver::~SparseQrLinearSystemSolver() {
      if (_factor) {
        _cholmod.free(_factor);
        _factor = NULL;
      }
    }


    void SparseQrLinearSystemSolver::initMatrixStructureImplementation(const std::vector<DesignVariable*>& dvs, const std::vector<ErrorTerm*>& errors, bool useDiagonalConditioner)
    {
      _errorTerms = errors;
      if (_factor) {
        _cholmod.free(_factor);
        _factor = NULL;
      }
      // should not be available or am i wrong?
      _useDiagonalConditioner = false; // useDiagonalConditioner;
      _jacobianBuilder.initMatrixStructure(dvs, errors);
      // spqr is only available with LONG indices
      CompressedColumnMatrix<SuiteSparse_long>& J_transpose = _jacobianBuilder.J_transpose();
      if (_useDiagonalConditioner) {
        J_transpose.pushConstantDiagonalBlock(1.0);
      }
      // View this matrix as a sparse matrix.
      // These views should remain valid for the lifetime of the object.
      J_transpose.getView(&_cholmodLhs);
      _cholmod.view(_e, &_cholmodRhs);
      if (_useDiagonalConditioner) {
        J_transpose.popDiagonalBlock();
      }
    }

    void SparseQrLinearSystemSolver::buildSystem(size_t nThreads, bool useMEstimator)
    {
      //std::cout << "build system\n";
      _jacobianBuilder.buildSystem(nThreads, useMEstimator);
      CompressedColumnMatrix<SuiteSparse_long>& J_transpose = _jacobianBuilder.J_transpose();
      J_transpose.rightMultiply(_e, _rhs);
      //std::cout << "build system complete\n";
    }

    bool SparseQrLinearSystemSolver::solveSystem(Eigen::VectorXd& outDx)
    {
      CompressedColumnMatrix<SuiteSparse_long>& J_transpose = _jacobianBuilder.J_transpose();
      if (_useDiagonalConditioner) {
        J_transpose.pushDiagonalBlock(_diagonalConditioner);
      }
      J_transpose.getView(&_cholmodLhs);
      _cholmod.view(_e, &_cholmodRhs);
      //std::cout << "solve system\n";
      if (!_factor) {
        //std::cout << "\tAnalyze system\n";
        // Now do the symbolic analysis with cholmod.
        _factor = _cholmod.analyzeQR(&_cholmodLhs);
        //std::cout << "\tanalyze system complete\n";
      }
      // Now we can solve the system.
      outDx.resize(J_transpose.rows());
      cholmod_dense* sol = _cholmod.solve(&_cholmodLhs, _factor, &_cholmodRhs,
        _options.qrTol, _options.colNorm);
      if (_useDiagonalConditioner) {
        J_transpose.popDiagonalBlock();
      }
      if (!sol) {
        std::cout << "Solution failed\n";
        return false;
      }
      try {
        SM_ASSERT_EQ_DBG(Exception, (int)sol->nrow, (int)outDx.size(), "Unexpected solution size");
        SM_ASSERT_EQ_DBG(Exception, sol->ncol, 1, "Unexpected solution size");
        SM_ASSERT_EQ_DBG(Exception, sol->xtype, (int)CholmodValueTraits<double>::XType, "Unexpected solution type");
        SM_ASSERT_EQ_DBG(Exception, sol->dtype, (int)CholmodValueTraits<double>::DType, "Unexpected solution type");
        memcpy((void*)&outDx[0], sol->x, sizeof(double)*sol->nrow);
      } catch (const Exception& e) {
        std::cout << e.what() << std::endl;
        // avoid leaking memory but still do error checking.
        // look at me! I done good.
        _cholmod.free(sol);
        throw;
      }
      _cholmod.free(sol);
      // std::cout << "solve system complete\n";
      return true;
    }

    double SparseQrLinearSystemSolver::getElement(const cholmod_sparse* R,
        size_t r, size_t c) {
      SM_ASSERT_LT_DBG(Exception, r, R->nrow, "Index out of bounds");
      SM_ASSERT_LT_DBG(Exception, c, R->ncol, "Index out of bounds");
      const SuiteSparse_long* row_ind =
        reinterpret_cast<const SuiteSparse_long*>(R->i);
      const SuiteSparse_long* col_ptr =
        reinterpret_cast<const SuiteSparse_long*>(R->p);
      const double* values =
        reinterpret_cast<const double*>(R->x);
      const SuiteSparse_long* colBegin = &row_ind[col_ptr[c]];
      const SuiteSparse_long* colEnd = &row_ind[col_ptr[c + 1]];
      const SuiteSparse_long* result = std::lower_bound(colBegin, colEnd, r);
      if (result != colEnd && *result == static_cast<SuiteSparse_long>(r))
        return values[result - &row_ind[0]];
      else
        return 0.0;
    }

    void SparseQrLinearSystemSolver::computeSigma(
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Sigma,
        size_t numCols) {
      CompressedColumnMatrix<SuiteSparse_long>& J_transpose =
        _jacobianBuilder.J_transpose();
      J_transpose.getView(&_cholmodLhs);
      _cholmod.view(_e, &_cholmodRhs);
      cholmod_sparse* R;
      SuiteSparse_long* E;
      _cholmod.getRE(&_cholmodLhs, &R, &E);
      Sigma = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::
        Zero(numCols, numCols);
      const size_t n = R->ncol;
      for (size_t l = n - 1, Sigma_l = numCols - 1;
          l >= n - numCols; --l, --Sigma_l) {
        double temp1 = 0;
        for (size_t j = l + 1, Sigma_j = Sigma_l + 1; j < n; ++j, ++Sigma_j) {
          temp1 += getElement(R, E[l], E[j]) * Sigma(Sigma_j, Sigma_l);
        }
        const double R_ll = getElement(R, E[l], E[l]);
        Sigma(Sigma_l, Sigma_l) = 1 / R_ll * (1 / R_ll - temp1);
        for (size_t i = l - 1, Sigma_i = Sigma_l - 1;
            i >= n - numCols; --i, --Sigma_i) {
          temp1 = 0;
          for (size_t j = i + 1, Sigma_j = Sigma_i + 1;
              j <= l; ++j, ++Sigma_j) {
            temp1 += getElement(R, E[i], E[j]) * Sigma(Sigma_j, Sigma_l);
          }
          double temp2 = 0;
          for (size_t j = l + 1, Sigma_j = Sigma_l + 1; j < n; ++j, ++Sigma_j) {
            temp2 += getElement(R, E[i], E[j]) * Sigma(Sigma_l, Sigma_j);
          }
          Sigma(Sigma_i, Sigma_l) = 1 / getElement(R, E[i], E[i]) *
            (-temp1 - temp2);
          Sigma(Sigma_l, Sigma_i) = Sigma(Sigma_i, Sigma_l);
        }
      }
      _cholmod.free(R);
      _cholmod.free(n, sizeof(SuiteSparse_long), E);
    }

    const SparseQRLinearSolverOptions&
    SparseQrLinearSystemSolver::getOptions() const {
      return _options;
    }

    SparseQRLinearSolverOptions&
    SparseQrLinearSystemSolver::getOptions() {
      return _options;
    }

    void SparseQrLinearSystemSolver::setOptions(
        const SparseQRLinearSolverOptions& options) {
      _options = options;
    }

  } // namespace backend
} // namespace aslam
