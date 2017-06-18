#ifndef ASLAM_BACKEND_BLOCK_CHOLESKY_LINEAR_SYSTEM_SOLVER_HPP
#define ASLAM_BACKEND_BLOCK_CHOLESKY_LINEAR_SYSTEM_SOLVER_HPP

#include "LinearSystemSolver.hpp"
#include <sparse_block_matrix/linear_solver.h>
#include <boost/shared_ptr.hpp>
#include "SparseBlockMatrixWrapper.hpp"

#include "aslam/backend/BlockCholeskyLinearSolverOptions.h"

namespace sm {

  class PropertyTree;

}
namespace aslam {
  namespace backend {

    class BlockCholeskyLinearSystemSolver : public LinearSystemSolver {
    public:
      typedef sparse_block_matrix::LinearSolver<Eigen::MatrixXd> LinearSolver;
      typedef sparse_block_matrix::SparseBlockMatrix<Eigen::MatrixXd> SparseBlockMatrix;

      BlockCholeskyLinearSystemSolver(const std::string & solver = "cholesky", const BlockCholeskyLinearSolverOptions& options= BlockCholeskyLinearSolverOptions());
      BlockCholeskyLinearSystemSolver(const sm::PropertyTree& config);
      ~BlockCholeskyLinearSystemSolver() override;


      /// \brief build the system of equations.
      void buildSystem(size_t nThreads, bool useMEstimator) override;

      /// \brief solve the system storing the solution in outDx and returning true on success.
      bool solveSystem(Eigen::VectorXd& outDx) override;

      /// \brief return the Hessian matrix if avaliable. Null if not available.
      const Matrix* Hessian() const override {
        return &_H;
      }

      std::string name() const override { return "block_" + _solverType; }

      /// \brief compute only the covariance blocks associated with the block indices passed as an argument
      void computeCovarianceBlocks(const std::vector<std::pair<int, int> >& blockIndices, SparseBlockMatrix& outP);

      void copyHessian(SparseBlockMatrix& H);

      /// Returns the options
      const BlockCholeskyLinearSolverOptions& getOptions() const;
      /// Returns the options
      BlockCholeskyLinearSolverOptions& getOptions();
      /// Sets the options
      void setOptions(const BlockCholeskyLinearSolverOptions& options);

      /// Helper Function for DogLeg implementation; returns parts required for the steepest descent solution
      double rhsJtJrhs() override;
        
    private:

      void initSolver();
      
      /// \brief initialized the matrix structure for the problem with these error terms and errors.
      void initMatrixStructureImplementation(const std::vector<DesignVariable*>& dvs, const std::vector<ErrorTerm*>& errors, bool useDiagonalConditioner) override;


      /// \brief The full Hessian matrix.
      SparseBlockMatrixWrapper _H;

      /// \brief the linear solver
      boost::shared_ptr<LinearSolver> _solver;

      /// Options
      BlockCholeskyLinearSolverOptions _options;

      std::string _solverType;
    };

  } // namespace backend
} // namespace aslam
#endif /* ASLAM_BACKEND_BLOCK_CHOLESKY_LINEAR_SYSTEM_SOLVER_HPP */
