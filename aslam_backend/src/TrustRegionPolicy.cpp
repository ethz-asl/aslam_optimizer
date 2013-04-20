#include <aslam/backend/TrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        TrustRegionPolicy::TrustRegionPolicy(Optimizer2Options & options) : _options(options){}
        TrustRegionPolicy::~TrustRegionPolicy(){}
            

        /// \brief get the linear system solver
        boost::shared_ptr<LinearSystemSolver> TrustRegionPolicy::getSolver()
        {
            return _solver;
        }

        /// \brief set the linear system solver
        void TrustRegionPolicy::setSolver(boost::shared_ptr<LinearSystemSolver> solver, Optimizer2Options options)
        {
            _solver = solver;
            _options = options;
        }
            
        bool TrustRegionPolicy::revertOnFailure()
        {
            return true;
        }

    } // namespace backend
} // namespace aslam
