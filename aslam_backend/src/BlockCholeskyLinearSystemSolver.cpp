#include <aslam/backend/BlockCholeskyLinearSystemSolver.hpp>
#include <sparse_block_matrix/linear_solver_cholmod.h>
#include <aslam/backend/ErrorTerm.hpp>

namespace aslam {
  namespace backend {
    BlockCholeskyLinearSystemSolver::BlockCholeskyLinearSystemSolver()
    {
      _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
    }

    BlockCholeskyLinearSystemSolver::~BlockCholeskyLinearSystemSolver()
    {
    }


    void BlockCholeskyLinearSystemSolver::initMatrixStructureImplementation(const std::vector<DesignVariable*>& dvs, const std::vector<ErrorTerm*>& errors, bool useDiagonalConditioner)
    {
      _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
      _useDiagonalConditioner = useDiagonalConditioner;
      _errorTerms = errors;
      std::vector<int> blocks;
      for (size_t i = 0; i < dvs.size(); ++i) {
        dvs[i]->setBlockIndex(i);
        blocks.push_back(dvs[i]->minimalDimensions());
      }
      std::partial_sum(blocks.begin(), blocks.end(), blocks.begin());
      // Now we can initialized the sparse Hessian matrix.
      _H._M = SparseBlockMatrix(blocks, blocks);
    }


    void BlockCholeskyLinearSystemSolver::buildSystem(size_t nThreads, bool useMEstimator)
    {
      // \todo make multithreaded. This is complicated as it requires synchronized access to the block matrix.
      //       A little bit of effort should make this possible by initializing the structure and adding
      //       a mutex for each block and having writers for each jacobian that have a list of mutexes.
      //       Save it for later.
      _H._M.clear(false);
      _rhs.setZero();
      std::vector<ErrorTerm*>::iterator it, it_end;
      it = _errorTerms.begin();
      it_end = _errorTerms.end();
      for (; it != it_end; ++it) {
        (*it)->buildHessian(_H._M, _rhs, useMEstimator);
      }
    }

    bool BlockCholeskyLinearSystemSolver::solveSystem(Eigen::VectorXd& outDx)
    {
      if (_useDiagonalConditioner) {
        Eigen::VectorXd d = _diagonalConditioner.cwiseProduct(_diagonalConditioner);
        // Augment the diagonal
        int rowBase = 0;
        for (int i = 0; i < _H._M.bRows(); ++i) {
          Eigen::MatrixXd& block = *_H._M.block(i, i, true);
          SM_ASSERT_EQ_DBG(Exception, block.rows(), block.cols(), "Diagonal blocks are square...right?");
          block.diagonal() += d.segment(rowBase, block.rows());
          rowBase += block.rows();
        }
      }
      // Solve the system
      outDx.resize(_H._M.rows());
      bool solutionSuccess = _solver->solve(_H._M, &outDx[0], &_rhs[0]);
      if (_useDiagonalConditioner) {
        // Un-augment the diagonal
        int rowBase = 0;
        for (int i = 0; i < _H._M.bRows(); ++i) {
          Eigen::MatrixXd& block = *_H._M.block(i, i, true);
          block.diagonal() -= _diagonalConditioner.segment(rowBase, block.rows());
          rowBase += block.rows();
        }
      }
      return solutionSuccess;
    }




    /// \brief compute only the covariance blocks associated with the block indices passed as an argument
    void BlockCholeskyLinearSystemSolver::computeCovarianceBlocks(const std::vector<std::pair<int, int> >& blockIndices, SparseBlockMatrix& outP)
    {
      // Not sure why I have to do this.
      //_solver->init();
      if (_useDiagonalConditioner) {
        Eigen::VectorXd d = _diagonalConditioner.cwiseProduct(_diagonalConditioner);
        // Augment the diagonal
        int rowBase = 0;
        for (int i = 0; i < _H._M.bRows(); ++i) {
          Eigen::MatrixXd& block = *_H._M.block(i, i, true);
          SM_ASSERT_EQ_DBG(Exception, block.rows(), block.cols(), "Diagonal blocks are square...right?");
          block.diagonal() += d.segment(rowBase, block.rows());
          rowBase += block.rows();
        }
      }
      bool success = _solver->solvePattern(outP, blockIndices, _H._M);
      SM_ASSERT_TRUE(Exception, success, "Unable to retrieve covariance");
      if (_useDiagonalConditioner) {
        // Un-augment the diagonal
        int rowBase = 0;
        for (int i = 0; i < _H._M.bRows(); ++i) {
          Eigen::MatrixXd& block = *_H._M.block(i, i, true);
          block.diagonal() -= _diagonalConditioner.segment(rowBase, block.rows());
          rowBase += block.rows();
        }
      }
    }

    void BlockCholeskyLinearSystemSolver::copyHessian(SparseBlockMatrix& H)
    {
      _H._M.cloneInto(H);
    }

    const BlockCholeskyLinearSolverOptions&
    BlockCholeskyLinearSystemSolver::getOptions() const {
      return _options;
    }

    BlockCholeskyLinearSolverOptions&
    BlockCholeskyLinearSystemSolver::getOptions() {
      return _options;
    }

    void BlockCholeskyLinearSystemSolver::setOptions(
        const BlockCholeskyLinearSolverOptions& options) {
      _options = options;
    }

  } // namespace backend
} // namespace aslam
