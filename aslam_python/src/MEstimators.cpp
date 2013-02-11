#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include <boost/shared_ptr.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
using namespace boost::python;
using namespace aslam::backend;


void exportMEstimators()
{

  class_< MEstimator, boost::shared_ptr<MEstimator>, boost::noncopyable >("MEstimator", no_init )
    .def("getWeight", &MEstimator::getWeight)
    .def("name", &MEstimator::name)
    ;

  class_< NoMEstimator, boost::shared_ptr<NoMEstimator>, bases<MEstimator> >("NoMEstimator", init<>())
  ;

  class_< GemanMcClureMEstimator, boost::shared_ptr<GemanMcClureMEstimator>, bases<MEstimator> >("GemanMcClureMEstimator", init<double>())
  ;

  class_< HuberMEstimator, boost::shared_ptr<HuberMEstimator>, bases<MEstimator> >("HuberMEstimator", init<double>())
  ;
         


}
