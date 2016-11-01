#ifndef ASLAM_BACKEND_OPTIMIZER_2_OPTIONS_HPP
#define ASLAM_BACKEND_OPTIMIZER_2_OPTIONS_HPP

#include <ostream>
#include <boost/shared_ptr.hpp>

#include <aslam/backend/OptimizerBase.hpp>

namespace aslam {
  namespace backend {
  class LinearSystemSolver;
  class TrustRegionPolicy;
  
    struct Optimizer2Options : public OptimizerOptionsBase {
      Optimizer2Options() :
        doSchurComplement(false),
        verbose(false),
        linearSolverMaximumFails(0)
      {
        convergenceDeltaError = 1e-3;
        convergenceDeltaX = 1e-3;
        maxIterations = 20;
      }

      /// \brief should we use the Schur complement trick? Currently not supported.
      bool doSchurComplement;

      /// \brief should we print out some information each iteration?
      bool verbose;

      /// \brief The number of times the linear solver may fail before the optimization is aborted. (>0 only if a fall back is available!)
      int linearSolverMaximumFails;

      boost::shared_ptr<LinearSystemSolver> linearSystemSolver;
      boost::shared_ptr<TrustRegionPolicy> trustRegionPolicy;
    };

    inline std::ostream& operator<<(std::ostream& out, const aslam::backend::Optimizer2Options& options)
    {
      out << static_cast<const OptimizerOptionsBase&>(options);
      out << "\tdoSchurComplement: " << options.doSchurComplement << std::endl;
      out << "\tverbose: " << options.verbose << std::endl;
      out << "\tlinearSolverMaximumFails: " << options.linearSolverMaximumFails << std::endl;
      return out;
    }
  } // namespace backend
}  // namespace aslam


#endif /* ASLAM_BACKEND_OPTIMIZER_OPTIONS_HPP */
