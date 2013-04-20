#ifndef ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP

#include <aslam/backend/TrustRegionPolicy.hpp>
#include <aslam/backend/LinearSystemSolver.hpp>
#include <boost/shared_ptr.hpp>

namespace aslam {
    namespace backend {
        
        class LevenbergMarquardtTrustRegionPolicy : public TrustRegionPolicy
        {
        public:
            LevenbergMarquardtTrustRegionPolicy(Optimizer2Options & options);
            virtual ~LevenbergMarquardtTrustRegionPolicy();
            
            /// \brief called by the optimizer when an optimization is starting
            virtual void optimizationStarting();
            
            // Returns true if the solution was successful
            virtual bool solveSystem(double J, bool previousIterationFailed, Eigen::VectorXd& outDx);
            
            /// \brief should the optimizer revert on failure? You should probably return true
            virtual bool revertOnFailure();
            
            /// \brief print the current state to a stream (no newlines).
            virtual std::ostream & printState(std::ostream & out);
            
        private:
            double _lambda;
        
            double getLmRho();
            
            double _J;
            double _p_J;
            double _gamma;
            double _beta;
            int _p;
            double _mu;
            
            Eigen::VectorXd _dx;
            
        };
        
    } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP */
