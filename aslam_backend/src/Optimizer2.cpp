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


namespace aslam {
  namespace backend {


    Optimizer2::Optimizer2(const Optimizer2Options& options) :
      _options(options)
    {
      initializeLinearSolver();
    }


    Optimizer2::~Optimizer2()
    {
    }


    /// \brief Set up to work on the optimization problem.
    void Optimizer2::setProblem(boost::shared_ptr<OptimizationProblemBase> problem)
    {
      _problem = problem;
    }


    void Optimizer2::initializeLinearSolver()
    {
      // \todo Add the remaining sparse_block_matrix solvers here.
      if (_options.linearSolver == "block_cholesky") {
        _options.verbose && std::cout << "Using the block cholesky linear solver.\n";
        _solver.reset(new BlockCholeskyLinearSystemSolver());
      } else if (_options.linearSolver == "sparse_cholesky") {
        _options.verbose && std::cout << "Using the sparse cholesky linear solver.\n";
        _solver.reset(new SparseCholeskyLinearSystemSolver());
      }
#ifndef QRSOLVER_DISABLED
      else if (_options.linearSolver == "sparse_qr") {
        _options.verbose && std::cout << "Using the sparse qr linear solver.\n";
        _solver.reset(new SparseQrLinearSystemSolver());
      }
#endif
      else if (_options.linearSolver == "dense_qr") {
        _options.verbose && std::cout << "Using the dense qr linear solver.\n";
        _solver.reset(new DenseQrLinearSystemSolver());
      } else {
        _options.verbose && std::cout << "Unknown linear solver specified: " << _options.linearSolver << ". Using the block cholesky linear solver.\n";
        _solver.reset(new BlockCholeskyLinearSystemSolver());
      }
    }

    /// \brief initialize the optimizer to run on an optimization problem.
    ///        This should be called before calling optimize()
    void Optimizer2::initialize()
    {
      SM_ASSERT_FALSE(Exception, _problem.get() == NULL, "No optimization problem has been set");
      _options.verbose && std::cout << "Initializing\n";
      Timer init("Optimizer2: Initialize Total");
      _designVariables.clear();
      _designVariables.reserve(_problem->numDesignVariables());
      _errorTerms.clear();
      _errorTerms.reserve(_problem->numErrorTerms());
      Timer initDv("Optimizer2: Initialize---Design Variables");
      // Run through all design variables adding active ones to an active list.
      // std::cout << "dvloop 1\n";
      for (size_t i = 0; i < _problem->numDesignVariables(); ++i) {
        DesignVariable* dv = _problem->designVariable(i);
        if (dv->isActive())
          _designVariables.push_back(dv);
      }
      SM_ASSERT_FALSE(Exception, _designVariables.empty(), "It is illegal to run the optimizer with all marginalized design variables.");
      // Assign block indices to the design variables.
      // "blocks" will hold the structure of the left-hand-side of Gauss-Newton
      int columnBase = 0;
      // std::cout << "dvloop 2\n";
      for (size_t i = 0; i < _designVariables.size(); ++i) {
        _designVariables[i]->setBlockIndex(i);
        _designVariables[i]->setColumnBase(columnBase);
        columnBase += _designVariables[i]->minimalDimensions();
      }
      initDv.stop();
      Timer initEt("Optimizer2: Initialize---Error Terms");
      // Get all of the error terms that work on these design variables.
      int dim = 0;
      // std::cout << "eloop 1\n";
      for (unsigned i = 0; i < _problem->numErrorTerms(); ++i) {
        ErrorTerm* e = _problem->errorTerm(i);
        _errorTerms.push_back(e);
        e->setRowBase(dim);
        dim += e->dimension();
      }
      initEt.stop();
      SM_ASSERT_FALSE(Exception, _errorTerms.empty(), "It is illegal to run the optimizer with no error terms.");
      Timer initMx("Optimizer2: Initialize---Matrices");
      // Set up the block matrix structure.
      // std::cout << "init structure\n";
//      initializeLinearSolver();
      _solver->initMatrixStructure(_designVariables, _errorTerms, _options.doLevenbergMarquardt);
      initMx.stop();
      _options.verbose && std::cout << "Optimization problem initialized with " << _designVariables.size() << " design variables and " << _errorTerms.size() << " error terms\n";
      // \todo Say how big the problem is.
      _options.verbose && std::cout << "The Jacobian matrix is " << dim << " x " << columnBase << std::endl;
    }

    double Optimizer2::getLmRho()
    {
      double d1 = _p_J - _J;    // update cost delta
      // L(0) - L(h):
      double d2 = _dx.transpose() * (_lambda * _dx + _solver->rhs());
      return d1 / d2;
    }


    void Optimizer2::setInitialLambda()
    {
      // check if _lambda already set:
      if (_lambda != 0) // if set return
        return;
      // \todo Reenable this.
      //if(_options.levenbergMarquardtEstimateLambdaScale <= 0 )
      {
        _lambda = _options.levenbergMarquardtLambdaInit;
      }
      // else
      // {
      //     double maximum = 0;
      //     // find the maximum diagonal element in H ans scale by LambdaScale:
      //     // loop the diagonal blocks:
      //     for(int i = 0; i < _H.bRows(); i++) {
      //         const Eigen::MatrixXd* H = _H.block(i,i,false); // do no allocate if missing!
      //         if (H) {
      //             for( int j = 0; j < H->rows(); j++) {
      //                 if (fabs((*H)(j,j)) > maximum)
      //                     maximum = fabs((*H)(i,j));
      //             }
      //         }
      //     }
      //     _lambda = maximum * _options.levenbergMarquardtEstimateLambdaScale;
      //     _options.verbose && std::cout << "Initialised Lambda: " << _lambda << std::endl;
      // }
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
    double abs_J = fabs(_J);

    // the first condition:
    bool crit1 = grad_norm < sqrt(epsilon) * (1 + abs_J);

    bool crit2 = _dx.norm() < sqrt(epsilon) * (1 + x_norm);

    bool crit3 = fabs(_J - _p_J) < epsilon * (1 + abs_J);

    bool crit4 = iterations < _options.maxIterations;

    return (crit1 && crit2 && crit3) || crit4;

    }*/




    /*
    SolutionReturnValue Optimizer2::optimizeDogLeg()
    {
        Timer timeGn("Optimizer2: build Hessian", true);
        Timer timeErr("Optimizer2: evaluate error", true);
        Timer timeSchur("Optimizer2: Schur complement", true);
        Timer timeBackSub("Optimizer2: Back substitution", true);
        Timer timeSolve("Optimizer2: Solve linear system", true);
        Timer timeSD("Optimizer2: Solve Steepest Descent", true);


        // Select the design variables and (eventually) the error terms involved in the optimization.
        initialize();

        SolutionReturnValue srv;
        _p_J = 0.0;

        // This sets _J
        evaluateError();

        _p_J = _J;
        srv.JStart = _p_J;

        // *** while not done
        _options.verbose && std::cout << "[" << srv.iterations << ".0]: J: " << _J << std::endl;

        // Set up the estimation problem.
        double deltaX = _options.convergenceDeltaX + 1.0;
        double deltaJ = _options.convergenceDeltaJ + 1.0;

        // choose initial delta to be something between GN and SD but biased towards GN
        // would be ideal but to avoid computing the GN solution in the first step before
        // looping we take the SD norm.
        double _delta = 0;
        double _p_delta = 0;
        // build Gauss newton matrices:
        timeGn.start();
        buildGnMatrices();
        timeGn.stop();

        Eigen::VectorXd _dx_sd(_H.rowBaseOfBlock(_marginalizedStartingBlock));
        Eigen::VectorXd _dx_gn(_H.rowBaseOfBlock(_marginalizedStartingBlock));

        double _dx_sd_norm = 0;
        double _dx_gn_norm = 0;
        double rho = 0;
        double L0 = 0;
        double _sd_scale = 0;
        double beta = 0;

        std::string stepType;

        // Loop until convergence
        while(   srv.iterations   < _options.maxIterations &&
                 deltaX       > _options.convergenceDeltaX &&
                 fabs(deltaJ)   > _options.convergenceDeltaJ)
            // while( !evaluateStoppingCriterion() )
        {
            // calculate steepest descent step:
            timeSD.start();

            Eigen::VectorXd Hrhs(_H.rows(),1);
            _H.multiply(&Hrhs, _rhs);

            _sd_scale = _rhs.squaredNorm() / Hrhs.squaredNorm();
            // std::cout << "  alpha:" << _sd_scale << std::endl;
            _dx_sd = _sd_scale * _rhs;

            timeSD.stop();

            // we need the norm for comparison:
            _dx_sd_norm = _dx_sd.norm();

            int stepIterations = 0;
            int linearSolverFailCounter = 0;

            bool gnComputed = false;
            do
            {
                // Trust Region smaller than SD:
                if (_dx_sd_norm >= _delta && _delta != 0)
                {
                    _dx = _delta / _dx_sd_norm * _dx_sd;  // scale SD step to fit into trust region
                    L0 = _delta * ( 2*_dx_sd_norm - _delta ) / ( 2*_sd_scale );
                    stepType = "SD";
                }
                // otherwise check the GN step
                else
                {
                    // calculate the GN step.
                    if(!gnComputed)
                    {
                        // set lambda to 0 in here...
                        _lambda = 0;

                        timeSchur.start();
                        applySchurComplement(_H, _rhs, _lambda,
                                             _marginalizedStartingBlock,
                                             _options.doLevenbergMarquardt,
                                             _A, _invVi, _b);
                        timeSchur.stop();

                        // Solve the dense system.
                        if(_options.resetSolverEveryIteration)
                        {
                            _solver->init();
                        }

                        timeSolve.start();
                        bool solutionSuccess = _solver->solve(_A, &_dx_gn[0], &_b[0]);
                        timeSolve.stop();

                        if(!solutionSuccess)
                        {

                            // the default solver failed. try the QR solver as a robust sparse alternative:
                            if(!_fallbackSolver) {  // initialise a new solver
                                // for now take QR
                                /// \todo: add an intelligent fallbackSolver structure.
                                /// maybe add it as a property of the linear solvers and directly name the solvers in their
                                /// class instead of the optimizer init()
                                _fallbackSolver.reset(new sparse_block_matrix::LinearSolverQr<Eigen::MatrixXd>());
                                _fallbackSolver->init();
                                _options.verbose && std::cout << "Optimizer2: The linear solution failed. Fallback to QR. (" << linearSolverFailCounter << ")" << std::endl;
                            }

                            solutionSuccess = _solver->solve(_A, &_dx_gn[0], &_b[0]);
                            if(!solutionSuccess) {
                                _options.verbose && std::cout << "Optimizer2: The linear solution failed. Fallback to QR. (" << linearSolverFailCounter << ")" << std::endl;
                                _dx_gn = _dx_sd;
                            }

                            // set the default solver to the robust one.
                            if(linearSolverFailCounter > _options.linearSolverMaximumFails)
                            {
                                _solver.reset(new sparse_block_matrix::LinearSolverQr<Eigen::MatrixXd>());
                                _solver->init();
                            }
                            linearSolverFailCounter++;  // increment
                            // SM_ASSERT_TRUE(Exception, linearSolverFailCounter <= _options.linearSolverMaximumFails, "The linear solution failed");

                        }
                        gnComputed = true;  // now we have it!
                        // and calculate the norm:
                        _dx_gn_norm = _dx_gn.norm();
                    }

                    // set delta in the first step to take a full GN step:
                    if(_delta == 0) {
                        _delta = (_dx_sd + 0.5 * ( _dx_gn - _dx_sd )).norm();
                    }
                    // now check the size of the gn step:
                    if(_dx_gn_norm <= _delta)
                    {
                        _dx = _dx_gn; // trust region larger than GN step. take it!
                        L0 = _J;
                        stepType = "GN";
                    }
                    else  // otherwise interpolate on the line between the cauchy point and gn step
                    {
                        // get beta:
                        Eigen::VectorXd dgnsd = _dx_gn - _dx_sd;
                        double gdnsd_norm_sqr = dgnsd.squaredNorm();

                        double c = _dx_sd.transpose() * ( dgnsd );
                        if(c <= 0)
                        {
                            beta = -c + sqrt( c*c + gdnsd_norm_sqr*(_delta*_delta - _dx_sd_norm*_dx_sd_norm) );
                            beta /= gdnsd_norm_sqr;
                        }
                        else
                        {
                            beta = (_delta*_delta - _dx_sd_norm*_dx_sd_norm);
                            beta /= c + sqrt( c*c + gdnsd_norm_sqr*(_delta*_delta - _dx_sd_norm*_dx_sd_norm) );
                        }

                        _dx = _dx_sd + beta * ( dgnsd );
                        L0 = 1/2 * _sd_scale * (1-beta)*(1-beta)* _rhs.squaredNorm() + beta*(2-beta)*_J;
                        stepType = "DL";
                    }
                }
                // update:
                timeBackSub.start();
                deltaX = applyStateUpdate();
                timeBackSub.stop();

                // reevaluate error
                timeErr.start();
                evaluateError();
                timeErr.stop();

                // same check as in GN lambda update:
                // rho = getLmRho();
                rho = (_p_J - _J) / L0;

                // std::cout << "RHO: " << rho << " L0: " << L0 << std::endl;
                if(rho > 0)
                {
                    // update GN matrices:
                    timeGn.start();
                    buildGnMatrices();
                    timeGn.stop();
                }
                else
                {
                    revertLastStateUpdate();
                }
                // update trust region
                _p_delta = _delta;
                if( rho > 0.75 ) // step succeeded
                {
                    double _dx_norm3 = 3 * _dx.norm();
                    if ( _delta > _dx_norm3 )
                        _delta = _delta;
                    else
                        _delta = _dx_norm3;
                }
                else if (rho > 0 && rho < 0.25) // step almost failed
                {
                    _delta /= 2.0;
                }
                else if (rho <= 0)  // step failed
                {
                    // if we took a GN step set the trust region to the GN Step / 2
                    if(stepType == "GN")
                        _delta = _dx_gn_norm / 2.0;
                    else
                        _delta /= 2.0;
                }

                deltaJ = _p_J - _J;
                if(_J < _p_J)
                    _p_J = _J;
                else
                    _J = _p_J;

                _options.verbose && std::cout << "   (" << stepIterations << "): J: " << _J << ", dJ: " << deltaJ << ", deltaX: " << deltaX << ", delta: " << _p_delta << ", StepType: " << stepType;

                if(stepType == "DL")
                    std::cout << ", beta: " << beta << std::endl;
                else
                    std::cout << std::endl;

                stepIterations++;

            } while( rho < 0 && _delta > _options.convergenceDeltaX );

            // count iterations
            srv.iterations++;

            _options.verbose && std::cout << "[" << srv.iterations << "]: J: " << _J << ", dJ: " << deltaJ << ", deltaX: " << deltaX << ", delta: " << _delta << std::endl << std::endl;

        }




        srv.JFinal = _J;
        srv.lmLambdaFinal = _lambda;
        srv.dXFinal = deltaX;
        srv.dJFinal = deltaJ;

        return srv;



    }

    */




    SolutionReturnValue Optimizer2::optimize()
    {
      Timer timeGn("Optimizer2: build Hessian", true);
      Timer timeErr("Optimizer2: evaluate error", true);
      Timer timeSchur("Optimizer2: Schur complement", true);
      Timer timeBackSub("Optimizer2: Back substitution", true);
      Timer timeSolve("Optimizer2: Solve linear system", true);
      // Select the design variables and (eventually) the error terms involved in the optimization.
      initialize();
      SolutionReturnValue srv;
      _p_J = 0.0;
      _lambda = 0;
      double gamma = _options.levenbergMarquardtLambdaGamma;
      double beta = _options.levenbergMarquardtLambdaBeta;
      int p = _options.levenbergMarquardtLambdaP;
      double mu = _options.levenbergMarquardtLambdaMuInit;
      //std::cout << "Evaluate error for the first time\n";
      // This sets _J
      timeErr.start();
      evaluateError(true);
      timeErr.stop();
      _p_J = _J;
      srv.JStart = _p_J;
      // *** while not done
      _options.verbose && std::cout << "[" << srv.iterations << ".0]: J: " << _J << std::endl;
      // Set up the estimation problem.
      double deltaX = _options.convergenceDeltaX + 1.0;
      double deltaJ = _options.convergenceDeltaJ + 1.0;
      bool isLmRegression = false;
      setInitialLambda();
      // Loop until convergence
      while (srv.iterations <  _options.maxIterations &&
             deltaX > _options.convergenceDeltaX &&
             fabs(deltaJ) > _options.convergenceDeltaJ) {
        // *** build: J, U, Vi, Wi, ea, ebi
        if (! isLmRegression) {
          timeGn.start();
          buildGnMatrices(true);
          timeGn.stop();
        }
        timeSolve.start();
        bool solutionSuccess = false;
        if (_options.doLevenbergMarquardt) {
          _solver->setConstantConditioner(_lambda);
          solutionSuccess = _solver->solveSystem(_dx);
        } else {
          solutionSuccess = _solver->solveSystem(_dx);
        }
        timeSolve.stop();
        if (!solutionSuccess) {
          _options.verbose && std::cout << "[WARNING] System solution failed\n";
          _options.verbose && std::cout << "          Reinitializing solver\n";
          _solver->initMatrixStructure(_designVariables, _errorTerms, _options.doLevenbergMarquardt);
          isLmRegression = true;
          // **** if J is a regression
          revertLastStateUpdate();
          _lambda *= mu;
          mu *= 2;
          srv.failedIterations++;
        } else {
          /// Apply the state update. _A, _b, _dx, and _H are passed in implicitly.
          timeBackSub.start();
          deltaX = applyStateUpdate();
          timeBackSub.stop();
          // This sets _J
          timeErr.start();
          evaluateError(true);
          timeErr.stop();
          deltaJ = _p_J - _J;
          if (_options.doLevenbergMarquardt) {
            // update rho:
            double rho = getLmRho();
            // if(_J > _p_J)
            if (rho <= 0) {
              isLmRegression = true;
              // **** if J is a regression
              revertLastStateUpdate();
              // ***** Update lambda
              // _lambda *= 1000;
              _lambda *= mu;
              mu *= 2;
              srv.failedIterations++;
            } else {
              isLmRegression = false;
              // **** otherwise
              // ***** Update lambda
              if (_lambda > 1e-16) {
                // _lambda *= 0.1;
                double u1 = 1 / gamma;
                double u2 = 1 - (beta - 1) * pow((2 * rho - 1), p);
                if (u1 > u2)
                  _lambda *= u1;
                else
                  _lambda *= u2;
                mu = _options.levenbergMarquardtLambdaBeta;
              } else
                _lambda = 1e-15;
            }
          }
        }
        srv.iterations++;
        if (_J < _p_J)
          _p_J = _J;
        _options.verbose && std::cout << "[" << srv.iterations << "]: J: " << _J << ", dJ: " << deltaJ << ", deltaX: " << deltaX << ", lambda: " << _lambda << std::endl;
        if (isLmRegression)
          _options.verbose && std::cout << "Regression in J. Reverting the last update\n";
      }
      srv.JFinal = _p_J;
      srv.lmLambdaFinal = _lambda;
      srv.dXFinal = deltaX;
      srv.dJFinal = deltaJ;
      return srv;
    }


    DesignVariable* Optimizer2::designVariable(size_t i)
    {
      SM_ASSERT_LT_DBG(Exception, i, _designVariables.size(), "index out of bounds");
      return _designVariables[i];
    }



    size_t Optimizer2::numDesignVariables() const
    {
      return _designVariables.size();
    }


    double Optimizer2::applyStateUpdate()
    {
      // Apply the update to the dense state.
      int startIdx = 0;
      for (size_t i = 0; i < numDesignVariables(); i++) {
        DesignVariable* d = _designVariables[i];
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
      for (size_t i = 0; i < _designVariables.size(); i++) {
        _designVariables[i]->revertUpdate();
      }
    }


    Optimizer2Options& Optimizer2::options()
    {
      return _options;
    }


    double Optimizer2::evaluateError(bool useMEstimator)
    {
      SM_ASSERT_TRUE(Exception, _solver.get() != NULL, "The solver is null");
      _J = _solver->evaluateError(_options.nThreads, useMEstimator);
      return _J;
    }

    void Optimizer2::buildGnMatrices(bool useMEstimator)
    {
      SM_ASSERT_TRUE(Exception, _solver.get() != NULL, "The solver is null");
      _solver->buildSystem(_options.nThreads, useMEstimator);
    }



    /// \brief return the reduced system dx
    const Eigen::VectorXd& Optimizer2::dx() const
    {
      return _dx;
    }

    /// The value of the objective function.
    double Optimizer2::J() const
    {
      return _J;
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
      std::vector<std::pair<int, int> > blockIndices;
      for (size_t i = 0; i < _designVariables.size(); ++i) {
        blockIndices.push_back(std::make_pair(i, i));
      }
      computeCovarianceBlocks(blockIndices, outP, lambda);
    }

    void Optimizer2::computeCovarianceBlocks(const std::vector<std::pair<int, int> > & blockIndices, SparseBlockMatrix& outP, double lambda)
    {
        SM_THROW(Exception, "Broken");
      BlockCholeskyLinearSystemSolver* solver = dynamic_cast<BlockCholeskyLinearSystemSolver*>(_solver.get());
      boost::shared_ptr<BlockCholeskyLinearSystemSolver> solver_sp;
      if (! solver || !_options.doLevenbergMarquardt) {
        _options.verbose && std::cout << "Creating a new block Cholesky solver to retrieve the covariance.\n";
        /// \todo Figure out properly why I have to create a new linear solver here.
        solver_sp.reset(new BlockCholeskyLinearSystemSolver());
        // True here for creating the diagonal conditioning.
        solver->initMatrixStructure(_designVariables, _errorTerms, true);
      }
      _options.verbose && std::cout << "Setting the diagonal conditioner to: " << lambda << ".\n";
      solver->setConstantConditioner(lambda);
      solver->buildSystem(_options.nThreads, false);
      solver->computeCovarianceBlocks(blockIndices, outP);
    }


    void Optimizer2::computeCovariances(SparseBlockMatrix& outP, double lambda)
    {
        SM_THROW(Exception, "Broken");

      std::vector<std::pair<int, int> > blockIndices;
      for (size_t i = 0; i < _designVariables.size(); ++i) {
        for (size_t j = i; j < _designVariables.size(); ++j) {
          blockIndices.push_back(std::make_pair(i, j));
        }
      }
      computeCovarianceBlocks(blockIndices, outP, lambda);
    }

    void Optimizer2::computeHessian(SparseBlockMatrix& outH, double lambda)
    {
 
      BlockCholeskyLinearSystemSolver* solver = dynamic_cast<BlockCholeskyLinearSystemSolver*>(_solver.get());
      boost::shared_ptr<BlockCholeskyLinearSystemSolver> solver_sp;
      if (! solver || !_options.doLevenbergMarquardt) {
        _options.verbose && std::cout << "Creating a new block Cholesky solver to retrieve the covariance.\n";
        /// \todo Figure out properly why I have to create a new linear solver here.
        solver_sp.reset(new BlockCholeskyLinearSystemSolver());
        // True here for creating the diagonal conditioning.
        solver->initMatrixStructure(_designVariables, _errorTerms, true);
      }
      _options.verbose && std::cout << "Setting the diagonal conditioner to: " << lambda << ".\n";
      solver->setConstantConditioner(lambda);
      solver->buildSystem(_options.nThreads, false);
      solver->copyHessian(outH);
    }

      const LinearSystemSolver * Optimizer2::getBaseSolver() const {
          return _solver.get();
      }


  } // namespace backend
} // namespace aslam
