#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/Frame.hpp>
#include <sm/python/Id.hpp>
#include <opencv2/core/eigen.hpp>
#include <aslam/cameras.hpp>

#include <aslam/BriskDescriptor.hpp>
#include <aslam/SurfDescriptor.hpp>
#include <aslam/python/ExportFrame.hpp>
using namespace boost::python;
using namespace aslam;


Eigen::MatrixXi getImage(FrameBase * frame)
{
  const cv::Mat & from = frame->image();
  Eigen::MatrixXi to(from.rows, from.cols);
  cv2eigen(from,to);

  return to;
}

void setImage(FrameBase * frame, const Eigen::MatrixXi & from)
{
  cv::Mat to; 
  eigen2cv(from,to);
  frame->setImage(to);
}

boost::shared_ptr<cameras::CameraGeometryBase> (FrameBase::*geometryBase)() = &FrameBase::geometryBase;

KeypointBase & (FrameBase::*keypointBase)(size_t i) = &FrameBase::keypointBase;

void exportFrame()
{
  sm::python::Id_python_converter<FrameId>::register_converter();
  sm::python::Id_python_converter<LandmarkId>::register_converter();

  class_<KeypointBase, boost::shared_ptr<KeypointBase>, boost::noncopyable>("KeypointBase", no_init)
    //const Time & time() const { return _time; }
    .def("time", &KeypointBase::time, return_value_policy<copy_const_reference>())
    //void setTime(const Time & time) { _time = time; }
    .def("setTime", &KeypointBase::setTime)
    //const sm::kinematics::UncertainHomogeneousPoint & landmark() const { return _landmark; }
    .def("landmark", &KeypointBase::landmark, return_value_policy<copy_const_reference>())
    //void setLandmark(const sm::kinematics::UncertainHomogeneousPoint & landmark){ _landmark = landmark; }
    .def("setLandmark", &KeypointBase::setLandmark)
    .def("landmarkId", &KeypointBase::landmarkId, return_value_policy<copy_const_reference>())
    //const LandmarkId & landmarkId() const { return _landmarkId; }
    .def("setLandmarkId", &KeypointBase::setLandmarkId)
    //void setLandmarkId(const LandmarkId & landmarkId){ _landmarkId = landmarkId; }
    .def("isLandmarkInitialized", &KeypointBase::isLandmarkInitialized)
    //bool isLandmarkInitialized() const { return _isLandmarkInitialized; }
    //void setLandmarkInitialized( bool isInitialized ) { _isLandmarkInitialized = isInitialized; }
    .def("setLandmarkInitialized", &KeypointBase::setLandmarkInitialized)
    //double homogeneousLandmarkSize() {return _homogeneousLandmarkSize;}
    .def("homogeneousLandmarkSize", &KeypointBase::homogeneousLandmarkSize)
    //void setHomogeneousLandmarkSize(double homogeneousLandmarkSize) {_homogeneousLandmarkSize=homogeneousLandmarkSize;}
    .def("setHomogeneousLandmarkSize", &KeypointBase::setHomogeneousLandmarkSize)
    ;

  class_<FrameBase, boost::shared_ptr<FrameBase>, boost::noncopyable>("FrameBase", no_init)
    .def("id", &FrameBase::id, return_value_policy<copy_const_reference>())
    .def("setId", &FrameBase::setId)
    .def("time", &FrameBase::time, return_value_policy<copy_const_reference>())
    .def("setTime", &FrameBase::setTime)
    .def("image", &getImage)
    .def("setImage", &setImage)
    .def("geometryBase", geometryBase)
    .def("keypointBase", keypointBase, return_internal_reference<>())
    .def("numKeypoints", &FrameBase::numKeypoints)
    // virtual Time keypointTime(size_t i) const = 0;
    ;

  using namespace aslam::cameras;
  aslam::python::exportFrame<PinholeCameraGeometry, BriskDescriptor>("PinholeBriskFrame");
  aslam::python::exportFrame<PinholeCameraGeometry, SurfDescriptor>("PinholeSurfFrame");

  aslam::python::exportFrame<DistortedPinholeCameraGeometry, BriskDescriptor>("DistortedPinholeBriskFrame");
  aslam::python::exportFrame<DistortedPinholeCameraGeometry, SurfDescriptor>("DistortedPinholeSurfFrame");

  aslam::python::exportFrame<PinholeRsCameraGeometry, SurfDescriptor>("PinholeRsSurfFrame");
  aslam::python::exportFrame<PinholeRsCameraGeometry, BriskDescriptor>("PinholeRsBriskFrame");


  aslam::python::exportFrame<DistortedPinholeRsCameraGeometry, SurfDescriptor>("DistortedPinholeRsSurfFrame");
  aslam::python::exportFrame<DistortedPinholeRsCameraGeometry, BriskDescriptor>("DistortedPinholeRsBriskFrame");
  aslam::python::exportCovarianceReprojectionError<DistortedPinholeRsCameraGeometry, BriskDescriptor>("DistortedPinholeRsBriskFrameCovarianceReprojectionError");

  aslam::python::exportFrame<OmniCameraGeometry, BriskDescriptor>("OmniBriskFrame");
  aslam::python::exportFrame<OmniCameraGeometry, SurfDescriptor>("OmniSurfFrame");
  
  // aslam::python::exportFrame<PinholeRsCameraDesignVariable, SurfDescriptor>("PinholeRsDVSurfFrame");
  // aslam::python::exportFrame<PinholeRsCameraDesignVariable, BriskDescriptor>("PinholeRsDVBriskFrame");
  
  // aslam::python::exportFrame<PinholeCameraDesignVariable, SurfDescriptor>("PinholeDVSurfFrame");
  // aslam::python::exportFrame<PinholeCameraDesignVariable, BriskDescriptor>("PinholeDVBriskFrame");

    
  aslam::python::exportKeypoint<2,BriskDescriptor>("Brisk");
  aslam::python::exportKeypoint<3,BriskDescriptor>("Brisk");
  aslam::python::exportKeypoint<4,BriskDescriptor>("Brisk");

  aslam::python::exportKeypoint<2,SurfDescriptor>("Surf");
  aslam::python::exportKeypoint<3,SurfDescriptor>("Surf");
  aslam::python::exportKeypoint<4,SurfDescriptor>("Surf");


  // export the optimizable intrinsics errors
  //aslam::python::exportReprojectionIntrinsicsError<PinholeRsCameraDesignVariable, SurfDescriptor>("PinholeRsDVSurfFrameReprojectionIntrinsicsError");
  //aslam::python::exportReprojectionIntrinsicsError<PinholeRsCameraDesignVariable, BriskDescriptor>("PinholeRsDVBriskFrameReprojectionIntrinsicsError");
  //aslam::python::exportReprojectionIntrinsicsError<PinholeCameraDesignVariable, SurfDescriptor>("PinholeDVSurfFrameReprojectionIntrinsicsError");
  //aslam::python::exportReprojectionIntrinsicsError<PinholeCameraDesignVariable, BriskDescriptor>("PinholeDVBriskFrameReprojectionIntrinsicsError");

}
