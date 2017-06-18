#ifndef ASLAM_BACKEND_DOG_LEG_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_DOG_LEG_TRUST_REGION_POLICY_HPP

#include <aslam/backend/TrustRegionPolicy.hpp>
#include <aslam/backend/LinearSystemSolver.hpp>
#include <boost/shared_ptr.hpp>

namespace aslam {
    namespace backend {
        
        class DogLegTrustRegionPolicy : public TrustRegionPolicy
        {
        public:
            DogLegTrustRegionPolicy();
            ~DogLegTrustRegionPolicy() override;
            
            /// \brief called by the optimizer when an optimization is starting
            void optimizationStartingImplementation(double J) override;
            
            // Returns true if the solution was successful
          bool solveSystemImplementation(double J, bool previousIterationFailed, int nThreads, Eigen::VectorXd& outDx) override;
            
            /// \brief should the optimizer revert on failure? You should probably return true
            bool revertOnFailure() override;
            
            /// \brief print the current state to a stream (no newlines).
            std::ostream & printState(std::ostream & out) const override;
          bool requiresAugmentedDiagonal() const override;
          std::string name() const override { return "dog_leg"; }
        private:
            
            Eigen::VectorXd _dx;
            Eigen::VectorXd _dx_sd;
            Eigen::VectorXd _dx_gn;
            
            double _dx_sd_norm;
            double _dx_gn_norm;
            double _L0;
            double _sd_scale;
            double _beta;
            double _delta;
            double _p_delta;
            std::string _stepType;
            
        };
        
    } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_DOG_LEG_TRUST_REGION_POLICY_HPP */
