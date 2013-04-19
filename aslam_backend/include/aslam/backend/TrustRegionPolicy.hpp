#ifndef ASLAM_BACKEND_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_TRUST_REGION_POLICY_HPP

#include <aslam/backend/LinearSystemSolver.hpp>
#include <boost/shared_ptr.hpp>

namespace aslam {
    namespace backend {
        
        class TrustRegionPolicy
        {
        public:
            TrustRegionPolicy();
            virtual ~TrustRegionPolicy();
            
            /// \brief called by the optimizer when an optimization is starting
            virtual void optimizationStarting() = 0;
            
            // Returns true if the solution was successful
            virtual bool solveSystem(Eigen::VectorXd& outDx, bool previousIterationFailed) = 0;

            /// \brief get the linear system solver
            boost::shared_ptr<LinearSystemSolver> getSolver();

            /// \brief set the linear system solver
            virtual void setSolver(boost::shared_ptr<LinearSystemSolver> solver) = 0;
            
            /// \brief should the optimizer revert on failure? You should probably return true
            virtual bool revertOnFailure();

            /// \brief print the current state to a stream (no newlines).
            virtual std::ostream & printState(std::ostream & out) = 0;
        protected:
            /// \brief the linear system solver.
            boost::shared_ptr<LinearSystemSolver> _solver;
                
        };

    } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_TRUST_REGION_POLICY_HPP */
