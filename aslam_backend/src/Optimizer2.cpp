#include <aslam/backend/Optimizer2.hpp>
// std::partial_sum
#include <numeric>
#include <aslam/backend/ErrorTerm.hpp>
// M.inverse()
#include <Eigen/Dense>
#include <sm/eigen/assert_macros.hpp>
#include <sparse_block_matrix/linear_solver_dense.h>
#include <sparse_block_matrix/linear_solver_cholmod.h>
#ifndef QRSOLVER_DISABLED
#include <sparse_block_matrix/linear_solver_spqr.h>
#include <aslam/backend/SparseQrLinearSystemSolver.hpp>
#endif
#include <aslam/backend/sparse_matrix_functions.hpp>
#include <aslam/backend/BlockCholeskyLinearSystemSolver.hpp>
#include <aslam/backend/SparseCholeskyLinearSystemSolver.hpp>
#include <aslam/backend/DenseQrLinearSystemSolver.hpp>
#include <sm/PropertyTree.hpp>


template <typename T>
T getDeprecatedPropertyIfItExists(const sm::ConstPropertyTree& config, const std::string & name, const std::string & newName, T defaultValue, T (sm::ConstPropertyTree::* getter)(const std::string & key, T defaultValue) const){
  const T depV = (config.*getter)(name, defaultValue);
  const T v = (config.*getter)(newName, defaultValue);
  if(depV != defaultValue){
    std::cerr << "Property " << name << " is DEPREACTED! Use " << newName << " instead." << std::endl;
    if(v != defaultValue){
      SM_THROW(std::runtime_error, "Both properties " + name + " (deprecated) and " + newName + " are used together!");
    }
    return depV;
  }
  return v;
}

namespace aslam {
    namespace backend {

        void Optimizer2::Status::resetImplementation() {
          srv = SolutionReturnValue();
        }

        Optimizer2::Optimizer2(const Options& options) :
            _options(options)
        {
            initializeLinearSolver();
            initializeTrustRegionPolicy();
        }

        Optimizer2::Optimizer2(const sm::ConstPropertyTree& config, boost::shared_ptr<LinearSystemSolver> linearSystemSolver, boost::shared_ptr<TrustRegionPolicy> trustRegionPolicy) {
          Options options;
          options.convergenceDeltaError = getDeprecatedPropertyIfItExists(config, "convergenceDeltaJ", "convergenceDeltaError", options.convergenceDeltaError, static_cast<double(sm::ConstPropertyTree::*)(const std::string&, double) const>(&sm::ConstPropertyTree::getDouble));
          options.convergenceDeltaX = config.getDouble("convergenceDeltaX", options.convergenceDeltaX);
          options.maxIterations = config.getInt("maxIterations", options.maxIterations);
          options.doSchurComplement = config.getBool("doSchurComplement", options.doSchurComplement);
          options.verbose = config.getBool("verbose", options.verbose);
          options.linearSolverMaximumFails = config.getInt("linearSolverMaximumFails", options.linearSolverMaximumFails);
          options.numThreadsJacobian = getDeprecatedPropertyIfItExists(config, "nThreads", "numThreadsJacobian", (int)options.numThreadsJacobian, static_cast<int(sm::ConstPropertyTree::*)(const std::string&, int) const>(&sm::ConstPropertyTree::getInt));
          options.numThreadsError = config.getInt("numThreadsError", options.numThreadsError);
          options.linearSystemSolver = linearSystemSolver;
          options.trustRegionPolicy = trustRegionPolicy;
          _options = options;
          initializeLinearSolver();
          initializeTrustRegionPolicy();
          // USING C++11 would allow to do constructor delegation and more elegant code, i.e., directly call the upper constructor
        }

        Optimizer2::~Optimizer2()
        {
        }

        void Optimizer2::initializeTrustRegionPolicy()
        {
          if( !_options.trustRegionPolicy ) {
            _options.verbose && std::cout << "No trust region policy set in the options. Defaulting to levenberg_marquardt\n";
            _trustRegionPolicy.reset( new LevenbergMarquardtTrustRegionPolicy() );
          } else {
            _trustRegionPolicy = _options.trustRegionPolicy;
          }


          // \todo remove this check when the sparse qr solver supports an augmented diagonal
          if(_solver->name() == "sparse_qr" && _trustRegionPolicy->name() == "levenberg_marquardt") {
            _options.verbose && std::cout << "The sparse_qr solver is not compatible with levenberg_marquardt. Changing to the dog_leg trust region policy\n";
            _trustRegionPolicy.reset( new DogLegTrustRegionPolicy() );
          }

          _options.verbose && std::cout << "Using the " << _trustRegionPolicy->name() << " trust region policy\n";

        }


        void Optimizer2::initializeLinearSolver()
        {
          if( ! _options.linearSystemSolver ) {
            _options.verbose && std::cout << "No linear system solver set in the options. Defaulting to the sparse_cholesky solver\n";
            _solver.reset(new SparseCholeskyLinearSystemSolver());
          } else {
            _solver = _options.linearSystemSolver;
          }

          _options.verbose && std::cout << "Using the " << _solver->name() << " linear system solver\n";
        }

        void Optimizer2::initializeImplementation()
        {
            OptimizerProblemManagerBase::initializeImplementation();
            initializeLinearSolver();
            initializeTrustRegionPolicy();

            Timer initMx("Optimizer2: Initialize---Matrices");
            // Set up the block matrix structure.
            _solver->initMatrixStructure(getDesignVariables(), problemManager().getErrorTerms(), _trustRegionPolicy->requiresAugmentedDiagonal());
            initMx.stop();
            _options.verbose && std::cout << "Optimization problem initialized with " << problemManager().numDesignVariables() << " design variables and " << problemManager().getErrorTerms().size() << " error terms\n";
            _options.verbose && std::cout << "The Jacobian matrix is " << problemManager().getTotalDimSquaredErrorTerms() << " x " << problemManager().numOptParameters() << std::endl;
        }


        /*
        // returns true of stop!
        bool Optimizer2::evaluateStoppingCriterion(int iterations)
        {

        // as we have analytic Jacobians we can assume the precision to be:
        double epsilon = std::numeric_limits<double>::epsilon();

        double x_norm = ...;

        // the gradient: is simply the right hand side of GN:
        double grad_norm = _rhs.norm();
        double abs_J = fabs(_status.error);

        // the first condition:
        bool crit1 = grad_norm < sqrt(epsilon) * (1 + abs_J);

        bool crit2 = _dx.norm() < sqrt(epsilon) * (1 + x_norm);

        bool crit3 = fabs(_status.error - _p_J) < epsilon * (1 + abs_J);

        bool crit4 = iterations < _options.maxIterations;

        return (crit1 && crit2 && crit3) || crit4;

        }*/

      SolutionReturnValue Optimizer2::optimize()
      {
        OptimizerProblemManagerBase::optimize();
        return _status.srv;
      }

        void Optimizer2::optimizeImplementation()
        {
            Timer timeErr("Optimizer2: evaluate error", true);
            Timer timeSchur("Optimizer2: Schur complement", true);
            Timer timeBackSub("Optimizer2: Back substitution", true);
            Timer timeSolve("Optimizer2: Build and solve linear system", true);
            // Select the design variables and (eventually) the error terms involved in the optimization.
            SolutionReturnValue & srv = _status.srv;
            _status.numIterations = srv.iterations;

            _p_J = -1.0;

            // This sets _J
            timeErr.start();
            evaluateError(true);
            timeErr.stop();
            _p_J = _status.error;
            srv.JStart = _p_J;
            // *** while not done
            _options.verbose && std::cout << "[" << srv.iterations << ".0]: J: " << _status.error << std::endl;
            // Set up the estimation problem.
            double & deltaX = _status.maxDeltaX;
            deltaX = _options.convergenceDeltaX + 1.0;
            double & deltaJ = _status.deltaError;
            deltaJ = _options.convergenceDeltaError + 1.0;
            bool previousIterationFailed = false;
            bool linearSolverFailure = false;

            SM_ASSERT_TRUE(Exception, _solver.get() != NULL, "The solver is null");
            _trustRegionPolicy->setSolver(_solver);
            _trustRegionPolicy->optimizationStarting(_status.error);

            issueCallback<callback::event::OPTIMIZATION_INITIALIZED>();

            // Loop until convergence
            while (srv.iterations <  _options.maxIterations &&
                   srv.failedIterations < _options.maxIterations &&
                   ((deltaX > _options.convergenceDeltaX &&
                     fabs(deltaJ) > _options.convergenceDeltaError) ||
                    linearSolverFailure)) {

                timeSolve.start();
                bool solutionSuccess = _trustRegionPolicy->solveSystem(_status.error, previousIterationFailed, _options.numThreadsError, _dx);
                SM_ASSERT_EQ(Exception, problemManager().numOptParameters(), size_t(_dx.size()), "_trustRegionPolicy->solveSystem yielded dx with wrong size!");
                timeSolve.stop();
                issueCallback<callback::event::LINEAR_SYSTEM_SOLVED>();

                if (!solutionSuccess) {
                    _options.verbose && std::cout << "[WARNING] System solution failed\n";
                    previousIterationFailed = true;
                    linearSolverFailure = true;
                    srv.failedIterations++;
                } else {
                    /// Apply the state update. _A, _b, _dx, and _H are passed in implicitly.
                    timeBackSub.start();
                    deltaX = applyStateUpdate();
                    timeBackSub.stop();
                    issueCallback<callback::event::DESIGN_VARIABLES_UPDATED>();
                    // This sets _J
                    timeErr.start();
                    evaluateError(true);
                    timeErr.stop();
                    deltaJ = _p_J - _status.error;
                    // This was a regression.
                    if( _trustRegionPolicy->revertOnFailure() )
                    {
                        if(deltaJ < 0.0)
                        {
                            _options.verbose && std::cout << "Last step was a regression. Reverting\n";
                            revertLastStateUpdate();
                            srv.failedIterations++;
                            previousIterationFailed = true;
                        }
                        else
                        {
                            _p_J = _status.error;
                            previousIterationFailed = false;
                        }
                    }
                    else
                    {
                        _p_J = _status.error;
                    }
                    srv.iterations++;
                    _status.numIterations = srv.iterations;

                    _options.verbose && std::cout << "[" << srv.iterations << "]: J: " << _status.error << ", dJ: " << deltaJ << ", deltaX: " << deltaX << ", ";
                    _options.verbose && _trustRegionPolicy->printState(std::cout);
                    _options.verbose && std::cout << std::endl;
                }
            } // if the linear solver failed / else
            srv.JFinal = _status.error = _p_J;
            srv.dXFinal = deltaX;
            srv.dJFinal = deltaJ;
            srv.linearSolverFailure = linearSolverFailure;

            //TODO make _status.convergence a set!
            if(srv.iterations >= _options.maxIterations){
              _status.convergence = MAX_ITERATIONS;
            } else if(linearSolverFailure || srv.failedIterations >= _options.maxIterations){
              _status.convergence = FAILURE;
            } else if (deltaX <= _options.convergenceDeltaX) {
              _status.convergence = DX;
            } else if (fabs(deltaJ) <= _options.convergenceDeltaError) {
              _status.convergence = DOBJECTIVE;
            }
        }


            DesignVariable* Optimizer2::designVariable(size_t i)
            {
                SM_ASSERT_LT_DBG(Exception, i, numDesignVariables(), "index out of bounds");
                return getDesignVariables().at(i);
            }



            size_t Optimizer2::numDesignVariables() const
            {
                return getDesignVariables().size();
            }


            double Optimizer2::applyStateUpdate()
            {
                // Apply the update to the dense state.
                int startIdx = 0;
                for (DesignVariable* d : getDesignVariables()) {
                    const int dbd = d->minimalDimensions();
                    Eigen::VectorXd dxS = _dx.segment(startIdx, dbd);
                    dxS *= d->scaling();
                    d->update(&dxS[0], dbd);
                    startIdx += dbd;
                }
                // Track the maximum delta
                // \todo: should this be some other metric?
                double deltaX = _dx.array().abs().maxCoeff();
                return deltaX;
            }





            void Optimizer2::revertLastStateUpdate()
            {
                for (DesignVariable * d : getDesignVariables()) {
                    d->revertUpdate();
                }
            }

            double Optimizer2::evaluateError(bool useMEstimator)
            {
              SM_ASSERT_TRUE(Exception, _solver.get() != NULL, "The solver is null");
              _status.error = _solver->evaluateError(_options.numThreadsError, useMEstimator, &_callbackManager);
              _callbackManager.issueCallback(callback::event::COST_UPDATED{_status.error, _p_J});
              return _status.error;
            }


            /// \brief return the reduced system dx
            const Eigen::VectorXd& Optimizer2::dx() const
            {
                return _dx;
            }

            /// The value of the objective function.
            double Optimizer2::J() const
            {
                return _status.error;
            }

            void Optimizer2::printTiming() const
            {
                sm::timing::Timing::print(std::cout);
            }







            void Optimizer2::checkProblemSetup()
            {
                // Check that all error terms are hooked up to design variables.
            }



            void Optimizer2::computeDiagonalCovariances(SparseBlockMatrix& outP, double lambda)
            {
                SM_THROW(Exception, "Broken");

                std::vector<std::pair<int, int> > blockIndices;
                for (size_t i = 0; i < getDesignVariables().size(); ++i) {
                    blockIndices.push_back(std::make_pair(i, i));
                }
                computeCovarianceBlocks(blockIndices, outP, lambda);
            }

    void Optimizer2::computeCovarianceBlocks(const std::vector<std::pair<int, int> > & /* blockIndices */, SparseBlockMatrix& /* outP */, double /* lambda */)
            {
                SM_THROW(Exception, "Broken");

            }


    void Optimizer2::computeCovariances(SparseBlockMatrix& /* outP */, double /* lambda */)
            {
                SM_THROW(Exception, "Broken");

            }

        void Optimizer2::computeHessian(SparseBlockMatrix& outH, double lambda)
            {

              boost::shared_ptr<BlockCholeskyLinearSystemSolver> solver_sp;
              solver_sp.reset(new BlockCholeskyLinearSystemSolver());
              // True here for creating the diagonal conditioning.
              solver_sp->initMatrixStructure(getDesignVariables(), problemManager().getErrorTerms(), true);

              _options.verbose && std::cout << "Setting the diagonal conditioner to: " << lambda << ".\n";
              evaluateError(false);
              solver_sp->setConstantConditioner(lambda);
              solver_sp->buildSystem(_options.numThreadsJacobian, false);
              solver_sp->copyHessian(outH);
            }

      const LinearSystemSolver * Optimizer2::getBaseSolver() const {
          return _solver.get();
      }



        const Matrix * Optimizer2::getJacobian() const {
            return _solver->Jacobian();
        }

        template <typename Event>
        void Optimizer2::issueCallback(){
          //TODO (HannesSommer) use ProceedInstruction value in the Optimizer
          _callbackManager.issueCallback(Event{_status.error, 0});
        }

        } // namespace backend
    } // namespace aslam
