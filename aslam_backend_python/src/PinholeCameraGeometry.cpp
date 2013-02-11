#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/cameras/PinholeCameraGeometry.hpp>

// add a wrapper for the distortion and undistortion functions:

namespace PinholeCameraGeometryWrapper {
    
//     // "Pass by reference" doesn't work with the Eigen type converters.
//     // So these functions must be wrapped. While we are at it, why 
//     // not make them nice.    
//     template<typename C>
//     boost::python::tuple distortion(const C * camera, double mx_u, double my_u)
//     {
//         double dx_u;
//         double dy_u;
//         Eigen::Matrix<double, 2, 1> p;
        
//         camera->distortion(mx_u, my_u, &dx_u, &dy_u);
        
//         p(0,0) = mx_u;
//         p(0,1) = my_u;
        
//         return boost::python::make_tuple(p);
//     }
    
//     template<typename C>
//     boost::python::tuple distortionAndJacobian(const C * camera, double mx_u, double my_u)
//     {
//         Eigen::MatrixXd H;
//         Eigen::Matrix<double, 2, 1> p;
        
//         double dx_u;
//         double dy_u;
//         double dxdmx; 
//         double dydmx;
//         double dxdmy; 
//         double dydmy;
        
//         camera->distortion(mx_u, my_u, &dx_u, &dy_u, &dxdmx, &dydmx, &dxdmy, &dydmy);
        
//         H(0,0) = dxdmx;
//         H(0,1) = dxdmy;
//         H(1,0) = dydmx;
//         H(1,1) = dydmy;
//         p(0,0) = mx_u;
//         p(0,1) = my_u;
        
//         return boost::python::make_tuple(p,H);
//     }
    
//     template<typename C>
//     boost::python::tuple undistortGN(const C * camera, double u_d, double v_d)
//     {
//         double u;
//         double v;
//         Eigen::Matrix<double, 2, 1> p;
        
//         camera->undistortGN(u_d, v_d, &u, &v);
        
//         p(0,0) = u;
//         p(0,1) = v;
        
//         return boost::python::make_tuple(p);
//     }
    
// }


void exportPinholeCameraGeometry()
{
  using namespace aslam::cameras;
  using namespace boost::python;
  //boost::python::class_<PinholeCameraGeometry, boost::shared_ptr<PinholeCameraGeometry>, boost::python::bases<CameraGeometryBase> >("PinholeCameraGeometry", "PinholeCameraGeometry(double focalLengthU, double focalLengthV, double imageCenterU, double imageCenterV, int resolutionU, int resolutionV)", boost::python::init<double,double,double,double,int,int>())
    
    //.def(boost::python::init<>())
    
}
