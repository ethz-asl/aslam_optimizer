#ifndef ASLAM_BACKEND_OPTIMIZER_RPROP_OPTIONS_HPP
#define ASLAM_BACKEND_OPTIMIZER_RPROP_OPTIONS_HPP

namespace aslam {
  namespace backend {
  class LinearSystemSolver;
  class TrustRegionPolicy;
  
    struct OptimizerRpropOptions {
    OptimizerRpropOptions() :
        etaMinus(0.5),
        etaPlus(1.2),
        initialDelta(0.1), // TODO: what is a good initial value?
        minDelta(1e-20),
        maxDelta(1.0),
        convergenceGradientNorm(1e-3),
        maxIterations(20),
        verbose(false),
//        linearSolverMaximumFails(0),
        nThreads(4)
      {

      }

      double etaMinus;
      double etaPlus;
      double initialDelta;
      double minDelta;
      double maxDelta;
      double convergenceGradientNorm;
//
      /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria.
      int maxIterations;

      /// \brief should we print out some information each iteration?
      bool verbose;

      /// \brief The number of threads to use
      int nThreads;

      boost::shared_ptr<LinearSystemSolver> linearSystemSolver;
      boost::shared_ptr<TrustRegionPolicy> trustRegionPolicy;
    };



    inline std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options)
    {
      /// \brief stop when steps cause changes in the objective function below this threshold.
      out << "OptimizerRpropOptions:\n";
      out << "\tetaMinus: " << options.etaMinus << std::endl;
      out << "\tetaPlus: " << options.etaPlus << std::endl;
      out << "\tinitialDelta: " << options.initialDelta << std::endl;
      out << "\tminDelta: " << options.minDelta << std::endl;
      out << "\tmaxDelta: " << options.maxDelta << std::endl;
      out << "\tmaxIterations: " << options.maxIterations << std::endl;
      out << "\tverbose: " << options.verbose << std::endl;
      out << "\tnThreads: " << options.nThreads << std::endl;
      return out;
    }

  } // namespace backend
} // namespace aslam
#endif /* ASLAM_BACKEND_OPTIMIZER_RPROP_OPTIONS_HPP */
