#include <aslam/backend/LevenbergMarquardtTrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        LevenbergMarquardtTrustRegionPolicy::LevenbergMarquardtTrustRegionPolicy(Optimizer2Options & options) : TrustRegionPolicy(options)  {}
        LevenbergMarquardtTrustRegionPolicy::~LevenbergMarquardtTrustRegionPolicy() {}
        
        
        /// \brief called by the optimizer when an optimization is starting
        void LevenbergMarquardtTrustRegionPolicy::optimizationStartingImplementation(double J)
        {
            // initialise lambda:
            _lambda = _options.levenbergMarquardtLambdaInit;
            _gamma = _options.levenbergMarquardtLambdaGamma;
            _beta = _options.levenbergMarquardtLambdaBeta;
            _p = _options.levenbergMarquardtLambdaP;
            _mu = _options.levenbergMarquardtLambdaMuInit;
            
        }
        
        // Returns true if the solution was successful
        bool LevenbergMarquardtTrustRegionPolicy::solveSystemImplementation(double J, bool previousIterationFailed, Eigen::VectorXd& outDx)
        {
            SM_ASSERT_TRUE(Exception, _solver.get() != NULL, "The solver is null");
            
            if (isFirstIteration()) {
                // This is the first step.
                _solver->buildSystem(_options.nThreads, true);
            } else {
                ///get Rho and update Lambda:
                double rho = getLmRho();
              
                if (rho <= 0) {
                    // The last step was a regression.
                    _lambda *= _mu;
                    _mu *= 2;
                    // No need to rebuild the system. Just reset the conditioner
                } else {
                    // The last iteration was successful
                    // Here we need to rebuild the system
                    _solver->buildSystem(_options.nThreads, true);
                    if (_lambda > 1e-16) {
                        double u1 = 1 / _gamma;
                        double u2 = 1 - (_beta - 1) * pow((2 * rho - 1), _p);
                        if (u1 > u2)
                            _lambda *= u1;
                        else
                            _lambda *= u2;
                        _mu = _options.levenbergMarquardtLambdaBeta;
                    } else {
                        _lambda = 1e-15;
                    }
                }
            }
            
            _solver->setConstantConditioner(_lambda);
            bool success = _solver->solveSystem(_dx);
            outDx = _dx;
            return success;
        }
        
        /// \brief print the current state to a stream (no newlines).
        std::ostream & LevenbergMarquardtTrustRegionPolicy::printState(std::ostream & out)
        {
            out << "LM - lambda:" << _lambda << " mu:" << _mu;
            return out;
        }

        
        bool LevenbergMarquardtTrustRegionPolicy::revertOnFailure()
        {
            return true;
        }
        
        
        double LevenbergMarquardtTrustRegionPolicy::getLmRho()
        {
            double d1 = get_dJ();    // update cost delta
            // L(0) - L(h):
            double d2 = _dx.transpose() * (_lambda * _dx + _solver->rhs());
            return d1 / d2;
        }
        
        
    } // namespace backend
} // namespace aslam
