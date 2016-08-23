#include <ctime>
#include <sm/eigen/gtest.hpp>
#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <sm/random.hpp>
#include "SampleDvAndError.hpp"
#include "TestOptimizerSupport.hpp"


TEST(OptimizerBFGSTestSuite, testBFGS)
{
  try {
    using namespace aslam::backend;
    sm::random::seed( std::time(NULL) );

    // Create linefit problem
    const Eigen::Vector2d paramTrue(1.0, 2.0);
    std::vector< boost::shared_ptr<TestNonSquaredError> > ets;
    Point2d dv;
    auto problem = test::createLinefitProblem(ets, dv, paramTrue, 1000, 0.1, 2.0);
    const double initialError = test::evaluateError(*problem);

    // Now let's optimize.
    OptimizerBFGS::Options options;
    options.maxIterations = 500;
    options.numThreadsGradient = 8;
    options.convergenceDeltaObjective = 0.0;
    options.convergenceDeltaX = 0.0;
    options.convergenceGradientNorm = 1e-9;
    OptimizerBFGS optimizer(options);
    optimizer.setProblem(problem);

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
    EXPECT_GT(ret.maxDeltaX, 0.0);
    EXPECT_EQ(test::evaluateError(*problem), ret.error);
    EXPECT_LT(ret.error, initialError);
    EXPECT_GT(ret.numIterations, 0);
    sm::eigen::assertNear(dv._v, paramTrue, 1e-1, SM_SOURCE_FILE_POS, "");

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
