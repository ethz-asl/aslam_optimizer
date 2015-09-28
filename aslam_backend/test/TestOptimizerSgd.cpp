#include <sm/eigen/gtest.hpp>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <aslam/backend/OptimizerSgd.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <sm/random.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

TEST(OptimizerStochasticGradientDescentTestSuite, testOptimizerStochasticGradientDescent)
{
  try {

    sm::random::seed(std::time(NULL));
    std::srand ( unsigned ( std::time(NULL) ) ); // for random_shuffle

    using namespace aslam::backend;

    auto normal_dist = [&] (int) { return sm::random::randn(); };

    // implement least squares line fit through data
    // Set up problem representing a linear function fit of the form y = a + b*x + N(0,1)
    // to noisy samples from a linear function
    const double a = 1.0;
    const double b = 2.0;
    const Eigen::Vector2d paramTrue(a, b);
    const double stddev = 0.1;
    const int numErrors = 1000;

    Point2d dv( paramTrue + 10.0*Eigen::Vector2d::NullaryExpr(normal_dist) ); // random initialization of design variable
    dv.setBlockIndex(0);
    dv.setActive(true);

    std::vector< boost::shared_ptr<TestNonSquaredError> > ets;
    ets.reserve(numErrors);
    for (std::size_t e = 0; e < numErrors; ++e) {
      const double x = (double)e/numErrors;
      const double y = a + b*x + stddev*sm::random::randn();
      boost::shared_ptr<TestNonSquaredError> err(new TestNonSquaredError(&dv, x, y));
      ets.push_back(err);
    }

    // Now let's optimize.
    OptimizerSgd::Options options;
    options.lambda0 = 1.0;
    options.tau = ets.size()/2.0;
    OptimizerSgd optimizer(options);

    OptimizerSgd::Status ret;
    std::size_t cnt = 0;
    for (size_t i=0; i<50; i++) {
      std::vector< boost::shared_ptr<TestNonSquaredError> > shuffledErrorTerms = ets;
      std::random_shuffle(shuffledErrorTerms.begin(), shuffledErrorTerms.end());
      for (auto& et : shuffledErrorTerms ) {
        SCOPED_TRACE("");
        optimizer.addBatch( std::vector<ScalarNonSquaredErrorTerm*>(1, et.get()) );
        optimizer.optimize();
        ret = optimizer.getStatus();
        SM_VERBOSE_STREAM("f(x) = " << dv._v[0] << " + " << dv._v[1] << " * x");
        cnt++;
        EXPECT_EQ(cnt, ret.numIterations);
      }
    }

    optimizer.incrementNumberOfIterations(2);
    EXPECT_EQ(50*ets.size()+2, ret.numIterations);

    EXPECT_LT(ret.learningRate, options.lambda0);
    EXPECT_GT(ret.learningRate, 0.0);
    EXPECT_GE(ret.maxDeltaX, 0.0);
    EXPECT_EQ(50*ets.size(), ret.numDerivativeEvaluations);

    EXPECT_NEAR(dv._v[0], a, 1e-1);
    EXPECT_NEAR(dv._v[1], b, 1e-1);

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
