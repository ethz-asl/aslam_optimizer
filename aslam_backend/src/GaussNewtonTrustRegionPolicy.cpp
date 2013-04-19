#include <aslam/backend/GaussNewtonTrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        
        GaussNewtonTrustRegionPolicy::GaussNewtonTrustRegionPolicy()  {}
        GaussNewtonTrustRegionPolicy::~GaussNewtonTrustRegionPolicy() {}
        
        
        /// \brief called by the optimizer when an optimization is starting
        void GaussNewtonTrustRegionPolicy::optimizationStarting()
        {
            
        }
        
        // Returns true if the solution was successful
        bool GaussNewtonTrustRegionPolicy::solveSystem(double J, bool previousIterationFailed, Eigen::VectorXd& outDx)
        {
            return _solver->solveSystem(outDx);
        }
        
        /// \brief print the current state to a stream (no newlines).
        std::ostream & GaussNewtonTrustRegionPolicy::printState(std::ostream & out)
        {
            
        }

        
        bool GaussNewtonTrustRegionPolicy::revertOnFailure()
        {
            return false;
        }
        
    } // namespace backend
} // namespace aslam
