#ifndef ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP

#include <aslam/backend/TrustRegionPolicy.hpp>
#include <aslam/backend/LinearSystemSolver.hpp>
#include <boost/shared_ptr.hpp>

namespace sm {
class ConstPropertyTree;
} // namespace sm

namespace aslam {
    namespace backend {
        
        class LevenbergMarquardtTrustRegionPolicy : public TrustRegionPolicy
        {
        public:
          LevenbergMarquardtTrustRegionPolicy();
          LevenbergMarquardtTrustRegionPolicy(const sm::ConstPropertyTree & config);
          LevenbergMarquardtTrustRegionPolicy(double lambdaInit);
          ~LevenbergMarquardtTrustRegionPolicy() override;

          /// \brief called by the optimizer when an optimization is starting
          void optimizationStartingImplementation(double J) override;

          // Returns true if the solution was successful
          bool solveSystemImplementation(double J, bool previousIterationFailed, int nThreads, Eigen::VectorXd& outDx) override;

          /// \brief should the optimizer revert on failure? You should probably return true
          bool revertOnFailure() override;

          /// \brief print the current state to a stream (no newlines).
          std::ostream & printState(std::ostream & out) const override;
          bool requiresAugmentedDiagonal() const override;
          std::string name() const override { return "levenberg_marquardt"; }
        private:
          double getLmRho(const Eigen::VectorXd & dx);
          double _lambdaInit;
          double _gammaInit;
          double _betaInit;
          int _pInit;
          double _muInit;
          
          double _lambda;
          double _gamma;
          double _beta;
          int _p;
          double _mu;
        };
        
    } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_LEVENBERG_MARQUARDT_TRUST_REGION_POLICY_HPP */
