#include <gtest/gtest.h>

#include <aslam/backend/ErrorTermObservationBST.hpp>
// This test harness makes it easy to test error terms.
#include <aslam/backend/test/ErrorTermTestHarness.hpp>
#include <sm/kinematics/Transformation.hpp>
#include <aslam/backend/TransformationBasic.hpp>

#include <aslam/backend/EuclideanPoint.hpp>
#include <sm/kinematics/RotationVector.hpp>
#include <aslam/splines/OPTBSpline.hpp>
#include <aslam/splines/implementation/OPTBSplineImpl.hpp>
#include <bsplines/EuclideanBSpline.hpp>
#include <aslam/backend/Scalar.hpp>

#include <boost/shared_ptr.hpp>

template <typename TConf, int ISplineOrder, int IDim, bool BDimRequired> struct ConfCreator {
  static inline TConf create(){
    return TConf(typename TConf::ManifoldConf(IDim), ISplineOrder);
  }
};

template <typename TConf, int ISplineOrder, int IDim> struct ConfCreator<TConf, ISplineOrder, IDim, false> {
  static inline TConf create(){
    BOOST_STATIC_ASSERT_MSG(IDim == TConf::Dimension::VALUE, "impossible dimension selected!");
    return TConf(typename TConf::ManifoldConf(), ISplineOrder);
  }
};

template <typename TConf, int ISplineOrder, int IDim> inline TConf createConf(){
  return ConfCreator<TConf, ISplineOrder, IDim, TConf::Dimension::IS_DYNAMIC>::create();
}


TEST(AslamVChargeBackendTestSuite, testEuclidean)
{
  try {
      using namespace aslam::backend;

      double sigma_n = 0.5;

      aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline robotPosSpline(createConf<bsplines::EuclideanBSpline<4, 1>::CONF, 4, 1>());
      const int pointSize = robotPosSpline.getPointSize();

      typename aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::point_t initPoint(pointSize);
      //typename aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::point_t p1(pointSize);

      initPoint(0,0) = 10.0;
      //p1(0,0) = 50;

      std::cout << "Init point is: " << std::endl << initPoint << std::endl;
      robotPosSpline.initConstantUniformSpline(0, 10, 10, initPoint);

      // First, create a design variable for the wall position.
      boost::shared_ptr<aslam::backend::Scalar> dv_w(new aslam::backend::Scalar(5.0));

    // Create observation error
    aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecPosExpr = robotPosSpline.getExpressionFactoryAt<1>(5).getValueExpression(0);
    aslam::backend::ErrorTermObservationBST eo(vecPosExpr, dv_w,  5.0, sigma_n * sigma_n);

    // Create the test harness
    aslam::backend::ErrorTermTestHarness<1> harness(&eo);

    // Run the unit tests.
    harness.testAll(1e-5);
  }
  catch(const std::exception & e)
    {
      FAIL() << e.what();
    }
}
