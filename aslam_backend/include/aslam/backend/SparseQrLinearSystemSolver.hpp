#ifndef ASLAM_BACKEND_SPARSE_QR_LINEAR_SYSTEM_SOLVER_HPP
#define ASLAM_BACKEND_SPARSE_QR_LINEAR_SYSTEM_SOLVER_HPP

#include "LinearSystemSolver.hpp"
#include "CompressedColumnJacobianTransposeBuilder.hpp"

#include "aslam/backend/SparseQRLinearSolverOptions.h"

namespace aslam {
  namespace backend {

    class SparseQrLinearSystemSolver : public LinearSystemSolver {
    public:
      typedef SuiteSparse_long index_t;

      SparseQrLinearSystemSolver();
      virtual ~SparseQrLinearSystemSolver();

      // virtual void evaluateError(size_t nThreads, bool useMEstimator);
      virtual void buildSystem(size_t nThreads, bool useMEstimator);
      virtual bool solveSystem(Eigen::VectorXd& outDx);
      // virtual void solveConstantAugmentedSystem(double diagonalConditioner, Eigen::VectorXd & outDx);
      // virtual void solveAugmentedSystem(const Eigen::VectorXd & diagonalConditioner, Eigen::VectorXd & outDx);

      /// Compute covariance block from the R matrix of QR (TO BE REMOVED)
      void computeSigma(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&
        Sigma, size_t numCols);

      /// Access an element of cholmod_sparse structure (TO BE REMOVED)
      double getElement(const cholmod_sparse* R, size_t r, size_t c);

      /// Returns the sum of log of the diagonal elements of the R matrix of QR (TO BE REMOVED)
      double computeSumLogDiagR(size_t numCols);

      /// Returns the current Jacobian transpose (note: const should be added)
      const CompressedColumnMatrix<index_t>& getJacobianTranspose();
      /// Returns the current estimated numerical rank
      index_t getRank() const;
      /// Returns the current tolerance
      double getTol() const;
      /// Returns the current permutation vector
      std::vector<index_t> getPermutationVector() const;
      /// Performs QR decomposition and returns the R matrix
      const CompressedColumnMatrix<index_t>& getR();
      /// Returns the current memory usage in bytes
      size_t getMemoryUsage() const;

      /// Returns the options
      const SparseQRLinearSolverOptions& getOptions() const;
      /// Returns the options
      SparseQRLinearSolverOptions& getOptions();
      /// Sets the options
      void setOptions(const SparseQRLinearSolverOptions& options);

    private:
      virtual void initMatrixStructureImplementation(const std::vector<DesignVariable*>& dvs, const std::vector<ErrorTerm*>& errors, bool useDiagonalConditioner);

      CompressedColumnJacobianTransposeBuilder<index_t> _jacobianBuilder;

      Cholmod<index_t> _cholmod;
      cholmod_sparse _cholmodLhs;
      cholmod_dense  _cholmodRhs;
#ifndef QRSOLVER_DISABLED
      SuiteSparseQR_factorization<double>* _factor;
      CompressedColumnMatrix<index_t> _R;
#endif
      SparseQRLinearSolverOptions _options;
    };

  } // namespace backend
} // namespace aslam
#endif /* ASLAM_BACKEND_SPARSE_QR_LINEAR_SYSTEM_SOLVER_HPP */
