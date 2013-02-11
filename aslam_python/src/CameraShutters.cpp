#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/cameras/RollingShutter.hpp>
#include <aslam/cameras/GlobalShutter.hpp>
#include <boost/serialization/nvp.hpp>
#include <aslam/Time.hpp>
#include <aslam/python/ExportDesignVariableAdapter.hpp>

using namespace boost::python;
using namespace aslam::cameras;
using namespace aslam;



template<typename S>
Duration temporalOffset(const S * shutter, const Eigen::VectorXd & keypoint)
{
	return shutter->temporalOffset(keypoint);
}



template<typename C, typename T>
void exportGenericShutterFunctions(T & shutter)
{
	shutter.def("temporalOffset", &temporalOffset<C>, "Returns the Temporal Offset of a given Keypoint");
    shutter.def("setParameters", &C::setParameters);

}


void exportGlobalShutter(std::string name) {

	class_<GlobalShutter, boost::shared_ptr<GlobalShutter > > globalShutter(name.c_str(), init<>());
		globalShutter.def(init<>((name + "()").c_str()))
			;
	exportGenericShutterFunctions<GlobalShutter>(globalShutter);
}


void exportRollingShutter(std::string name) {

	class_<RollingShutter, boost::shared_ptr<RollingShutter > > rollingShutter(name.c_str(), init<>());
	rollingShutter.def(init<>((name + "()").c_str()))
		.def(init<double>((name + "(double lineDelay)").c_str()))
		.def("lineDelay",&RollingShutter::lineDelay, "Returns the Line Delay of the Rolling Shutter")
		;

	exportGenericShutterFunctions<RollingShutter>(rollingShutter);
}



void exportCameraShutters() {

	  using namespace boost::python;
	  using namespace aslam::cameras;

	  exportGlobalShutter("GlobalShutter");
	  exportRollingShutter("RollingShutter");

	  aslam::python::exportDesignVariableAdapter< GlobalShutter >("GlobalShutterDesignVariable");
	  aslam::python::exportDesignVariableAdapter< RollingShutter >("RollingShutterDesignVariable");

}
