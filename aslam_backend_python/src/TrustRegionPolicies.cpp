#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/TrustRegionPolicy.hpp>
#include <aslam/backend/GaussNewtonTrustRegionPolicy.hpp>
#include <aslam/backend/LevenbergMarquardtTrustRegionPolicy.hpp>
#include <aslam/backend/DogLegTrustRegionPolicy.hpp>
#include <aslam/backend/LineSearchTrustRegionPolicy.hpp>


using namespace boost::python;
using namespace aslam::backend;

void exportTrustRegionPolicies() {

  class_<TrustRegionPolicy, boost::shared_ptr<TrustRegionPolicy>, boost::noncopyable>("TrustRegionPolicy", no_init)
      .def("name", &TrustRegionPolicy::name)
      .def("requiresAugmentedDiagonal", &TrustRegionPolicy::requiresAugmentedDiagonal)
      ;

  // GN
  class_<GaussNewtonTrustRegionPolicy, boost::shared_ptr<GaussNewtonTrustRegionPolicy>, bases< TrustRegionPolicy >, boost::noncopyable >("GaussNewtonTrustRegionPolicy", init<>() )
      ;
  
  // LM
  class_<LevenbergMarquardtTrustRegionPolicy, boost::shared_ptr<LevenbergMarquardtTrustRegionPolicy>, bases< TrustRegionPolicy >, boost::noncopyable >("LevenbergMarquardtTrustRegionPolicy", init<>() )
      .def(init<double>("LevenbergMarquardtTrustRegionPolicy( double initalLambda )"))
      ;

  // DL
  class_<DogLegTrustRegionPolicy, boost::shared_ptr<DogLegTrustRegionPolicy>, bases< TrustRegionPolicy >, boost::noncopyable >("DogLegTrustRegionPolicy", init<>() )
      ;
  
  // LS
  class_<LineSearchTrustRegionPolicy, boost::shared_ptr<LineSearchTrustRegionPolicy>, bases< TrustRegionPolicy >, boost::noncopyable >("LineSearchTrustRegionPolicy", init<>())
      .def(init<double>("LineSearchTrustRegionPolicy( double scaleDxOnFailure )"))
      .def("setScaleStep", &LineSearchTrustRegionPolicy::setScaleStep)
      .def("getScaleStep", &LineSearchTrustRegionPolicy::getScaleStep)
          ;

}
