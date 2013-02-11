#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/cameras/OmniCameraGeometry.hpp>

void exportOmniCameraGeometry()
{
    using namespace aslam::cameras;
    boost::python::class_<OmniCameraGeometry, boost::shared_ptr<OmniCameraGeometry>, boost::python::bases<CameraGeometryBase> >("OmniCameraGeometry", "OmniCameraGeometry(double xi, double k1, double k2, double p1, double p2, double gamma1, double gamma2, double u0, double v0, int width, int height)", boost::python::init<double,double,double,double,double,double,double,double,double, int, int>())
      .def(boost::python::init<>())
      .def("createTestGeometry", &OmniCameraGeometry::createTestGeometry, "Create a sample pinhole camera")
      .staticmethod("createTestGeometry")
      //.def("saveBinary",&OmniCameraGeometry::saveBinary,"Save the camera structure as a boost binary archive")
      //.def("loadBinary",&OmniCameraGeometry::loadBinary, "Load the camera structure from a boost binary archive")
      //.def("saveXml",&OmniCameraGeometry::saveXml,"Save the camera structure as a boost xml archive")
      //.def("loadXml",&OmniCameraGeometry::loadXml,"Load the camera structure as a boost xml archive")
      .def("width",&OmniCameraGeometry::width,"The width of an image")
      .def("height",&OmniCameraGeometry::height,"The height of an image")
      .def("cols",&OmniCameraGeometry::width,"The width of an image")
      .def("rows",&OmniCameraGeometry::height,"The height of an image")

      //.def("",&OmniCameraGeometry::,"")
      //.def("",&OmniCameraGeometry::,"")
      ;

      // \todo

      // void undistortGN(double u_d, double v_d, double * u, double * v) const;
      // void distortion(double mx_u, double my_u, double *mx_d, double *my_d) const;
      // // Functions from Chris Mei's library
      // void distortion(double mx_u, double my_u, double *mx_d, double *my_d,
      // 		      double *dxdmx, double *dydmx,
      // 		      double *dxdmy, double *dydmy) const;
      // // Lift points from the image plane to the sphere
      // void lift_sphere(double u, double v, double *X, double *Y, double *Z) const;
      
      // // Lift points from the image plane to the projective space
      // void lift_projective(double u, double v, double *X, double *Y, double *Z) const;
      
      // // Projects 3D points to the image plane (Pi function)
      // void space2plane(double x, double y, double z, double *u, double *v) const;
      
      // // Projects 3D points to the image plane (Pi function)
      // // and calculates jacobian
      // void space2plane(double x, double y, double z,
      // 		       double *u, double *v,
      // 		       double *dudx, double *dvdx,
      // 		       double *dudy, double *dvdy,
      // 		       double *dudz, double *dvdz) const;
      
      // void undist2plane(double mx_u, double my_u, double *u, double *v) const;


      
      // /// \brief intrinsic parameter xi.
      // double xi() const { return _xi; }
      
      // /// \brief intrinsic distortion parameter k1
      // double k1() const { return _k1; }
      // /// \brief intrinsic distortion parameter k2
      // double k2() const { return _k2; }
      // /// \brief intrinsic distortion parameter p1
      // double p1() const { return _p1; }
      // /// \brief intrinsic distortion parameter p2
      // double p2() const { return _p2; }
      // /// \brief intrinsic distortion parameter gamma1
      // double gamma1() const { return _gamma1; }
      // /// \brief intrinsic distortion parameter gamma2
      // double gamma2() const { return _gamma2; }
      // /// \brief intrinsic distortion parameter u0
      // double u0() const { return _u0; }
      // /// \brief intrinsic distortion parameter v0
      // double v0() const { return _v0; }
      // /// \brief The horizontal resolution in pixels.
      // int width() const { return _width; }
      // int cols() const { return _width; }
      // /// \brief The vertical resolution in pixels.
      // int height() const { return _height; }
      // int rows() const { return _height; }

      // double inv_K11() const { return _inv_K11; }
      // double inv_K13() const { return _inv_K13; }
      // double inv_K22() const { return _inv_K22; }
      // double inv_K23() const { return _inv_K23; }
      // double one_over_xixi_m_1() const { return _one_over_xixi_m_1; }

}
