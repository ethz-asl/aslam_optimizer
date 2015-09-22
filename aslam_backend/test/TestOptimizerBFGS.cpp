#include <sm/eigen/gtest.hpp>
#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <sm/random.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

TEST(OptimizerBFGSTestSuite, testBFGS)
{
  try {
    using namespace aslam::backend;
    boost::shared_ptr<OptimizationProblem> problem_ptr(new OptimizationProblem);
    OptimizationProblem& problem = *problem_ptr;

    sm::random::seed( std::time(NULL) );
    auto normal_dist = [&] (int) { return sm::random::randn(); };

    // implement least squares line fit through data
    // Set up problem representing a linear function fit of the form y = a + b*x + N(0,1)
    // to noisy samples from a linear function
    const double a = 1.0;
    const double b = 2.0;
    const Eigen::Vector2d paramTrue(a, b);
    const double stddev = 0.1;
    const int numErrors = 1000;

    Point2d dv( paramTrue + 2.0*Eigen::Vector2d::NullaryExpr(normal_dist) ); // random initialization of design variable
    dv.setBlockIndex(0);
    dv.setColumnBase(0);
    dv.setActive(true);
    problem.addDesignVariable(&dv, false);

    std::vector< boost::shared_ptr<TestNonSquaredError> > ets;
    ets.reserve(numErrors);
    for (std::size_t e = 0; e < numErrors; ++e) {
      const double x = (double)e/numErrors;
      const double y = a + b*x + stddev*sm::random::randn();
      boost::shared_ptr<TestNonSquaredError> err(new TestNonSquaredError(&dv, x, y));
      testErrorTerm(err);
      ets.push_back(err);
      problem.addErrorTerm(err);
    }

    // Now let's optimize.
    OptimizerBFGS::Options options;
    options.maxIterations = 500;
    options.numThreadsGradient = 8;
    options.convergenceDeltaObjective = 0.0;
    options.convergenceGradientNorm = 0.0;
    options.convergenceDeltaX = 0.0;
    EXPECT_ANY_THROW(options.check());
    options.convergenceGradientNorm = 1e-15;
    EXPECT_NO_THROW(options.check());
    OptimizerBFGS optimizer(options);
    optimizer.setProblem(problem_ptr);

    // Test that linesearch options are correctly forwarded
    options.linesearch.initialStepLength = 1.1;
    optimizer.setOptions(options);
    EXPECT_DOUBLE_EQ(options.linesearch.initialStepLength, optimizer.getLineSearch().options().initialStepLength);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    optimizer.initialize();
    SCOPED_TRACE("Optimizing");
    optimizer.optimize();
    const auto& ret = optimizer.getStatus();

    EXPECT_GT(ret.convergence, ConvergenceStatus::FAILURE);
    EXPECT_LE(ret.gradientNorm, options.convergenceGradientNorm);
    EXPECT_GT(ret.numObjectiveEvaluations, 0);
    EXPECT_GT(ret.numDerivativeEvaluations, 0);
    EXPECT_GE(ret.error, 0.0);
    EXPECT_LT(ret.deltaError, 1e-12);
    EXPECT_LT(ret.maxDeltaX, 1e-3);
    EXPECT_LT(ret.error, std::numeric_limits<double>::max());
    EXPECT_GT(ret.numIterations, 0);
    EXPECT_NEAR(dv._v[0], a, 1e-1);
    EXPECT_NEAR(dv._v[1], b, 1e-1);

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
