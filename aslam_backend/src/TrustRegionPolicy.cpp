#include <aslam/backend/TrustRegionPolicy.hpp>

namespace aslam {
    namespace backend {
        
        TrustRegionPolicy::TrustRegionPolicy(){}
        TrustRegionPolicy::~TrustRegionPolicy(){}
            

        /// \brief get the linear system solver
        boost::shared_ptr<LinearSystemSolver> TrustRegionPolicy::getSolver()
        {
            return _solver;
        }

        /// \brief set the linear system solver
        void TrustRegionPolicy::setSolver(boost::shared_ptr<LinearSystemSolver> solver)
        {
            _solver = solver;
        }
            


    } // namespace backend
} // namespace aslam
