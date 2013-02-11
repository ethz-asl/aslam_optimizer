#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/DesignVariable.hpp>
#include <aslam/cameras/PinholeRSCameraDesignVariable.hpp>
#include <aslam/cameras/PinholeCameraDesignVariable.hpp>

// add a wrapper for the distortion and undistortion functions:
void exportPinholeCameraDesignVariable()
{
  using namespace aslam::cameras;
  //boost::python::class_<PinholeCameraGeometry, boost::shared_ptr<PinholeCameraGeometry>, boost::python::bases<CameraGeometryBase> >("PinholeCameraGeometry", "PinholeCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV)", boost::python::init<double,double,double,double,int,int>())
    
    //.def(boost::python::init<>())
    
    boost::python::class_<PinholeCameraDesignVariable, boost::shared_ptr<PinholeCameraDesignVariable>, boost::python::bases<PinholeCameraGeometry, aslam::backend::DesignVariable> >("PinholeDesignVariable", boost::python::init<double, double, double, double, int, int, double, double, double, double>("PinholeCameraDesignVariable(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double k1, double k2, double p1, double p2)")) // boost::python::init<double,double,double,double,int,int>()
   // .def(boost::python::init<double, double, double, double, int, int, double, double, double, double>("PinholeCameraDesignVariable(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double k1, double k2, double p1, double p2)"))
 /*   .def(boost::python::init<double, double, double, double, int, int, double, double, double, double>("PinholeCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, k1, k2, p1, p2)"))
    .def("focalLengthCol",&PinholeCameraGeometry::focalLengthCol, "The focal length corresponding to the columns (horizontal)")
    .def("focalLengthRow",&PinholeCameraGeometry::focalLengthRow, "The focal length corresponding to the rows (vertical)")
    .def("opticalCenterCol",&PinholeCameraGeometry::opticalCenterCol, "The horizontal optical center of the camera (in pixels)")
    .def("opticalCenterRow",&PinholeCameraGeometry::opticalCenterRow, "The vertical optical center of the camera (in pixels)")
    .def("width",&PinholeCameraGeometry::width, "The width of an image")
    .def("height",&PinholeCameraGeometry::height, "The height of an image")
    .def("cols",&PinholeCameraGeometry::width, "The width of an image")
    .def("rows",&PinholeCameraGeometry::height, "The height of an image")
    .def("k1",&PinholeCameraGeometry::k1, "Radial Distortion Parameter 1")
    .def("k2",&PinholeCameraGeometry::k2, "Radial Distortion Parameter 2")
    .def("p1",&PinholeCameraGeometry::p1, "Distortion Parameter 1")
    .def("p2",&PinholeCameraGeometry::p2, "Distortion Parameter 2")
    .def("createTestGeometry", &PinholeCameraGeometry::createTestGeometry, "Create a sample pinhole camera")
    .def("createDistortedTestGeometry", &PinholeCameraGeometry::createDistortedTestGeometry, "Create a sample pinhole camera with distortions")
    .staticmethod("createTestGeometry")
    .staticmethod("createDistortedTestGeometry")
    .def("distortion", &PinholeCameraGeometryWrapper::distortion<PinholeCameraGeometry>,"Get Distortion.")
    .def("distortionAndJacobian", &PinholeCameraGeometryWrapper::distortionAndJacobian<PinholeCameraGeometry>,"Get Distortion and Jacobian.")
    .def("undistortGN", &PinholeCameraGeometryWrapper::undistortGN<PinholeCameraGeometry>,"Undistort a point with Least Squares solver.")
    */
  ;
}

void exportPinholeRSCameraDesignVariable()
{
  using namespace aslam::cameras;
  //boost::python::class_<PinholeCameraGeometry, boost::shared_ptr<PinholeCameraGeometry>, boost::python::bases<CameraGeometryBase> >("PinholeCameraGeometry", "PinholeCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV)", boost::python::init<double,double,double,double,int,int>())

    //.def(boost::python::init<>())

    boost::python::class_<PinholeRSCameraDesignVariable, boost::shared_ptr<PinholeRSCameraDesignVariable>, boost::python::bases<PinholeRSCameraGeometry, aslam::backend::DesignVariable> >("PinholeRSDesignVariable", boost::python::init<double, double, double, double, int, int, double, double, double, double, double>("PinholeRSCameraDesignVariable(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double lineDelay, double k1, double k2, double p1, double p2)")) // boost::python::init<double,double,double,double,int,int>()
   // .def(boost::python::init<double, double, double, double, int, int, double, double, double, double>("PinholeCameraDesignVariable(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double k1, double k2, double p1, double p2)"))
 /*   .def(boost::python::init<double, double, double, double, int, int, double, double, double, double>("PinholeCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, k1, k2, p1, p2)"))
    .def("focalLengthCol",&PinholeCameraGeometry::focalLengthCol, "The focal length corresponding to the columns (horizontal)")
    .def("focalLengthRow",&PinholeCameraGeometry::focalLengthRow, "The focal length corresponding to the rows (vertical)")
    .def("opticalCenterCol",&PinholeCameraGeometry::opticalCenterCol, "The horizontal optical center of the camera (in pixels)")
    .def("opticalCenterRow",&PinholeCameraGeometry::opticalCenterRow, "The vertical optical center of the camera (in pixels)")
    .def("width",&PinholeCameraGeometry::width, "The width of an image")
    .def("height",&PinholeCameraGeometry::height, "The height of an image")
    .def("cols",&PinholeCameraGeometry::width, "The width of an image")
    .def("rows",&PinholeCameraGeometry::height, "The height of an image")
    .def("k1",&PinholeCameraGeometry::k1, "Radial Distortion Parameter 1")
    .def("k2",&PinholeCameraGeometry::k2, "Radial Distortion Parameter 2")
    .def("p1",&PinholeCameraGeometry::p1, "Distortion Parameter 1")
    .def("p2",&PinholeCameraGeometry::p2, "Distortion Parameter 2")
    .def("createTestGeometry", &PinholeCameraGeometry::createTestGeometry, "Create a sample pinhole camera")
    .def("createDistortedTestGeometry", &PinholeCameraGeometry::createDistortedTestGeometry, "Create a sample pinhole camera with distortions")
    .staticmethod("createTestGeometry")
    .staticmethod("createDistortedTestGeometry")
    .def("distortion", &PinholeCameraGeometryWrapper::distortion<PinholeCameraGeometry>,"Get Distortion.")
    .def("distortionAndJacobian", &PinholeCameraGeometryWrapper::distortionAndJacobian<PinholeCameraGeometry>,"Get Distortion and Jacobian.")
    .def("undistortGN", &PinholeCameraGeometryWrapper::undistortGN<PinholeCameraGeometry>,"Undistort a point with Least Squares solver.")
    */
  ;
}
