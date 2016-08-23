#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/../../../test/SampleDvAndError.hpp>

#include <boost/shared_ptr.hpp>


using namespace boost::python;
using namespace aslam::backend;

Eigen::Vector2d getValue(const Point2d& p) {
  return p._v;
}

void exportSampleDvAndError()
{

    class_<Point2d, boost::shared_ptr<Point2d>, bases<DesignVariable> >
      ("Point2d", init<const Eigen::Vector2d&>("Point2d(vector2d v): Constructor"))
      .add_property("_v", &getValue)
      ;

    class_<LinearErr, boost::shared_ptr<LinearErr>, bases< aslam::backend::ErrorTermFs<2> > >
      ("LinearErr", init<Point2d*>("LinearErr(Point2d p): Constructor"))
      ;

    class_<LinearErr2, boost::shared_ptr<LinearErr2>, bases< aslam::backend::ErrorTermFs<2> > >
      ("LinearErr2", init<Point2d*,Point2d*>("LinearErr2(Point2d p0, Point2d p1): Constructor"))
      ;

    class_<LinearErr3, boost::shared_ptr<LinearErr3>, bases< aslam::backend::ErrorTermFs<4> > >
      ("LinearErr3", init<Point2d*,Point2d*,Point2d*>("LinearErr3(Point2d p0, Point2d p1, Point2d p2): Constructor"))
      ;

    class_<TestNonSquaredError, boost::shared_ptr<TestNonSquaredError>, bases<ScalarNonSquaredErrorTerm> >
      ("TestNonSquaredError", init<Point2d*, const double, const double>("TestNonSquaredError(Point2d p, double x, double y): Constructor"))
      .def_readwrite("_y", &TestNonSquaredError::_y)
      .def_readwrite("_dv", &TestNonSquaredError::_dv)
      ;

}
