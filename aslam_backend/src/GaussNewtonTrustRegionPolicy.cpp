#include <aslam/backend/GaussNewtonTrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        
        GaussNewtonTrustRegionPolicy::GaussNewtonTrustRegionPolicy(Optimizer2Options & options) : TrustRegionPolicy(options)  {}
        GaussNewtonTrustRegionPolicy::~GaussNewtonTrustRegionPolicy() {}
        
        
        /// \brief called by the optimizer when an optimization is starting
        void GaussNewtonTrustRegionPolicy::optimizationStarting()
        {
            
        }
        
        // Returns true if the solution was successful
        bool GaussNewtonTrustRegionPolicy::solveSystem(double J, bool previousIterationFailed, Eigen::VectorXd& outDx)
        {
            _solver->buildSystem(_options.nThreads, true);
            return _solver->solveSystem(outDx);
        }
        
        /// \brief print the current state to a stream (no newlines).
        std::ostream & GaussNewtonTrustRegionPolicy::printState(std::ostream & out)
        {
            return out;
        }

        
        bool GaussNewtonTrustRegionPolicy::revertOnFailure()
        {
            return false;
        }
        
    } // namespace backend
} // namespace aslam
