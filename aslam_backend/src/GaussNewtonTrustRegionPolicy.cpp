#include <aslam/backend/GaussNewtonTrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        
        /// \brief called by the optimizer when an optimization is starting
        void GaussNewtonTrustRegionPolicy::optimizationStarting()
        {
            
        }
        
        // Returns true if the solution was successful
        bool GaussNewtonTrustRegionPolicy::solveSystem(Eigen::VectorXd& outDx, bool previousIterationFailed)
        {
            return _solver->solveSystem(outDx);
        }
        
        /// \brief print the current state to a stream (no newlines).
        std::ostream & printState(std::ostream & out)
        {
            
        }

        
        bool GaussNewtonTrustRegionPolicy::revertOnFailure()
        {
            return false;
        }
        
    } // namespace backend
} // namespace aslam
