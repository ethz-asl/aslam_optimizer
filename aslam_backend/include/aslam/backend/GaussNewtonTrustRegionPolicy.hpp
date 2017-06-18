#ifndef ASLAM_BACKEND_GAUSS_NEWTON_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_GAUSS_NEWTON_TRUST_REGION_POLICY_HPP

#include <aslam/backend/TrustRegionPolicy.hpp>
#include <aslam/backend/LinearSystemSolver.hpp>
#include <boost/shared_ptr.hpp>

namespace aslam {
    namespace backend {
        
        class GaussNewtonTrustRegionPolicy : public TrustRegionPolicy
        {
        public:
            GaussNewtonTrustRegionPolicy();
            ~GaussNewtonTrustRegionPolicy() override;
            
            /// \brief called by the optimizer when an optimization is starting
            void optimizationStartingImplementation(double J) override;
            
            // Returns true if the solution was successful
          bool solveSystemImplementation(double J, bool previousIterationFailed, int nThreads, Eigen::VectorXd& outDx) override;
            
            /// \brief should the optimizer revert on failure? You should probably return true
            bool revertOnFailure() override;
            
            /// \brief print the current state to a stream (no newlines).
            std::ostream & printState(std::ostream & out) const override;
          std::string name() const override { return "gauss_newton"; }
          bool requiresAugmentedDiagonal() const override;

        
            
        };
        
    } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_GAUSS_NEWTON_TRUST_REGION_POLICY_HPP */
