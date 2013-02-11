#ifndef ASLAM_BACKEND_OPTIMIZER_2_OPTIONS_HPP
#define ASLAM_BACKEND_OPTIMIZER_2_OPTIONS_HPP

namespace aslam {
  namespace backend {

    struct Optimizer2Options {
      Optimizer2Options() :
        convergenceDeltaJ(1e-3),
        convergenceDeltaX(1e-3),
        maxIterations(20),
        levenbergMarquardtLambdaInit(1e-3),
        levenbergMarquardtLambdaGamma(3),
        levenbergMarquardtLambdaBeta(2),
        levenbergMarquardtLambdaP(3),
        levenbergMarquardtLambdaMuInit(2),
        levenbergMarquardtEstimateLambdaScale(-1),
        doLevenbergMarquardt(true),
        doSchurComplement(false),
        verbose(false),
        linearSolverMaximumFails(0),
        linearSolver("block_cholesky"),
        nThreads(4)


      {};

      /// \brief stop when steps cause changes in the objective function below this threshold.
      double convergenceDeltaJ;

      /// \brief stop when the maximum change in and component of a design variable drops below this threshold
      double convergenceDeltaX;

      /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria.
      int maxIterations;

      /// \brief what value of lambda (for Levenberg-Marquardt) should be used when initializing the optimization
      double levenbergMarquardtLambdaInit;

      /// \brief the parameters required for levenberg marquard Lambda updates:
      double levenbergMarquardtLambdaGamma;
      int levenbergMarquardtLambdaBeta;     // odd!
      int levenbergMarquardtLambdaP;        // odd!
      double levenbergMarquardtLambdaMuInit;

      /// \brief negative values indicate that the LamdaInit inital value should be used
      double levenbergMarquardtEstimateLambdaScale;

      /// \brief should we use Levenberg-Marquardt? Otherwise use plain Gauss-Newton.
      bool doLevenbergMarquardt;

      /// \brief should we use the Schur complement trick? Currently not supported.
      bool doSchurComplement;

      /// \brief should we print out some information each iteration?
      bool verbose;

      /// \brief The number of times the linear solver may fail before the optimisation is aborted. (>0 only if a fallback is available!)
      int linearSolverMaximumFails;

      /// \brief which linear solver should we use. Options are currently "block_cholesky", "sparse_cholesky", "sparse_qr".
      std::string linearSolver;

      /// \brief The number of threads to use
      int nThreads;
    };



    inline std::ostream& operator<<(std::ostream& out, const aslam::backend::Optimizer2Options& options)
    {
      /// \brief stop when steps cause changes in the objective function below this threshold.
      out << "Optimizer2Options:\n";
      out << "\tconvergenceDeltaJ: " << options.convergenceDeltaJ << std::endl;
      /// \brief stop when the maximum change in and component of a design variable drops below this threshold
      out << "\tconvergenceDeltaX: " << options.convergenceDeltaX << std::endl;
      /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria.
      out << "\tmaxIterations: " << options.maxIterations << std::endl;
      /// \brief what value of lambda (for Levenberg-Marquardt) should be used when initializing the optimization
      out << "\tlevenbergMarquardtLambdaInit: " << options.levenbergMarquardtLambdaInit << std::endl;
      /// \brief the parameters required for levenberg marquard Lambda updates:
      out << "\tlevenbergMarquardtLambdaGamma: " << options.levenbergMarquardtLambdaGamma << std::endl;
      out << "\tlevenbergMarquardtLambdaBeta: " << options.levenbergMarquardtLambdaBeta << std::endl;     // odd!
      out << "\tlevenbergMarquardtLambdaP: " << options.levenbergMarquardtLambdaP << std::endl;        // odd!
      out << "\tlevenbergMarquardtLambdaMuInit: " << options.levenbergMarquardtLambdaMuInit << std::endl;
      /// \brief negative values indicate that the LamdaInit inital value should be used
      out << "\tlevenbergMarquardtEstimateLambdaScale: " << options.levenbergMarquardtEstimateLambdaScale << std::endl;
      /// \brief should we use Levenberg-Marquardt? Otherwise use plain Gauss-Newton.
      out << "\tdoLevenbergMarquardt: " <<  options.doLevenbergMarquardt << std::endl;
      /// \brief should we use the Schur complement trick? Currently not supported.
      out << "\tdoSchurComplement: " << options.doSchurComplement << std::endl;
      /// \brief should we print out some information each iteration?
      out << "\tverbose: " << options.verbose << std::endl;
      /// \brief The number of times the linear solver may fail before the optimisation is aborted. (>0 only if a fallback is available!)
      out << "\tlinearSolverMaximumFails: " << options.linearSolverMaximumFails << std::endl;
      /// \brief which linear solver should we use. Options are currently "block_cholesky", "sparse_cholesky", "sparse_qr".
      out << "\tlinearSolver: " << options.linearSolver << std::endl;
      /// \brief The number of threads to use
      out << "\tnThreads: " << options.nThreads << std::endl;
      return out;
    }

  } // namespace backend
} // namespace aslam
#endif /* ASLAM_BACKEND_OPTIMIZER_OPTIONS_HPP */
