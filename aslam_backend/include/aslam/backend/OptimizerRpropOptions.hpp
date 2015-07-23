#ifndef ASLAM_BACKEND_OPTIMIZER_RPROP_OPTIONS_HPP
#define ASLAM_BACKEND_OPTIMIZER_RPROP_OPTIONS_HPP

namespace aslam {
  namespace backend {

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
        nThreads(4)
      {

      }

      double etaMinus; /// \brief Decrease factor for step size if gradient direction changes
      double etaPlus; /// \brief Increase factor for step size if gradient direction is same
      double initialDelta; /// \brief Initial step size
      double minDelta; /// \brief Minimum step size
      double maxDelta; /// \brief Maximum step size
      double convergenceGradientNorm; /// \brief Stopping criterion on gradient norm
      int maxIterations; /// \brief stop if we reach this number of iterations without hitting any of the above stopping criteria.
      bool verbose; /// \brief should we print out some information each iteration?
      int nThreads; /// \brief The number of threads to use

    };

    inline std::ostream& operator<<(std::ostream& out, const aslam::backend::OptimizerRpropOptions& options)
    {
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
