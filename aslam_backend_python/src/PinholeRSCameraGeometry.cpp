#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/cameras/PinholeRSCameraGeometry.hpp>


namespace PinholeRSCameraGeometryWrapper {
    
    // "Pass by reference" doesn't work with the Eigen type converters.
    // So these functions must be wrapped. While we are at it, why 
    // not make them nice.    
    template<typename C>
    boost::python::tuple distortion(const C * camera, double mx_u, double my_u)
    {
        double dx_u;
        double dy_u;
        Eigen::Matrix<double, 2, 1> p;
        
        camera->distortion(mx_u, my_u, &dx_u, &dy_u);
        
        p(0,0) = mx_u;
        p(0,1) = my_u;
        
        return boost::python::make_tuple(p);
    }
    
    template<typename C>
    boost::python::tuple distortionAndJacobian(const C * camera, double mx_u, double my_u)
    {
        Eigen::MatrixXd H;
        Eigen::Matrix<double, 2, 1> p;
        
        double dx_u;
        double dy_u;
        double dxdmx; 
        double dydmx;
        double dxdmy; 
        double dydmy;
        
        camera->distortion(mx_u, my_u, &dx_u, &dy_u, &dxdmx, &dydmx, &dxdmy, &dydmy);
        
        H(0,0) = dxdmx;
        H(0,1) = dxdmy;
        H(1,0) = dydmx;
        H(1,1) = dydmy;
        p(0,0) = mx_u;
        p(0,1) = my_u;
        
        return boost::python::make_tuple(p,H);
    }
    
    template<typename C>
    boost::python::tuple undistortGN(const C * camera, double u_d, double v_d)
    {
        double u;
        double v;
        Eigen::Matrix<double, 2, 1> p;
        
        camera->undistortGN(u_d, v_d, &u, &v);
        
        p(0,0) = u;
        p(0,1) = v;
        
        return boost::python::make_tuple(p);
    }
    
}


void exportPinholeRSCameraGeometry()
{
  using namespace aslam::cameras;
  boost::python::class_<PinholeRSCameraGeometry, boost::shared_ptr<PinholeRSCameraGeometry>, boost::python::bases<CameraGeometryBase> >("PinholeRSCameraGeometry", boost::python::init<>())
    
    .def(boost::python::init<double, double, double, double, int, int, double>("PinholeTSCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double lineDelay)"))
    .def(boost::python::init<double, double, double, double, int, int,double, double, double, double, double>("PinholeTSCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV, double lineDelay, k1, k2, p1, p2)"))  
    
    
    .def("focalLengthCol",&PinholeRSCameraGeometry::focalLengthCol, "The focal length corresponding to the columns (horizontal)")
    .def("focalLengthRow",&PinholeRSCameraGeometry::focalLengthRow, "The focal length corresponding to the rows (vertical)")
    .def("opticalCenterCol",&PinholeRSCameraGeometry::opticalCenterCol, "The horizontal optical center of the camera (in pixels)")
    .def("opticalCenterRow",&PinholeRSCameraGeometry::opticalCenterRow, "The vertical optical center of the camera (in pixels)")
    .def("width",&PinholeRSCameraGeometry::width, "The width of an image")
    .def("height",&PinholeRSCameraGeometry::height, "The height of an image")
    .def("cols",&PinholeRSCameraGeometry::width, "The width of an image")
    .def("rows",&PinholeRSCameraGeometry::height, "The height of an image")
    .def("k1",&PinholeRSCameraGeometry::k1, "Radial Distortion Parameter 1")
    .def("k2",&PinholeRSCameraGeometry::k2, "Radial Distortion Parameter 2")
    .def("p1",&PinholeRSCameraGeometry::p1, "Distortion Parameter 1")
    .def("p2",&PinholeRSCameraGeometry::p2, "Distortion Parameter 2")
    .def("lineDelay",&PinholeRSCameraGeometry::lineDelay, "The delay between the start of integration of two consecutive lines")
    .def("createTestGeometry", &PinholeRSCameraGeometry::createTestGeometry, "Create a sample pinhole camera")
    .staticmethod("createTestGeometry")
    
    .def("distortion", &PinholeRSCameraGeometryWrapper::distortion<PinholeRSCameraGeometry>,"Get Distortion.")
    .def("distortionAndJacobian", &PinholeRSCameraGeometryWrapper::distortionAndJacobian<PinholeRSCameraGeometry>,"Get Distortion and Jacobian.")
    .def("undistortGN", &PinholeRSCameraGeometryWrapper::undistortGN<PinholeRSCameraGeometry>,"Undistort a point with Least Squares solver.")
    
    
  ;
}
