#include <ctime>
#include <sm/eigen/gtest.hpp>
#include <sm/random.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"
#include "TestOptimizerSupport.hpp"


TEST(OptimizerRpropTestSuite, testOptions)
{
  try {
    using namespace aslam::backend;
    OptimizerRprop::Options options;
    TEST_INVALID_OPTION(options, etaMinus, 0.0);
    TEST_INVALID_OPTION(options, etaPlus, 0.0);
    TEST_INVALID_OPTION(options, initialDelta, 0.0);
    TEST_INVALID_OPTION(options, minDelta, 0.0);
    TEST_INVALID_OPTION(options, maxDelta, 0.0);
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerRpropTestSuite, testRpropNonSquaredErrorTerms)
{
  try {
    using namespace aslam::backend;
    sm::random::seed( std::time(NULL) );

    // Create linefit problem
    const Eigen::Vector2d paramTrue(1.0, 2.0);
    std::vector< boost::shared_ptr<TestNonSquaredError> > ets;
    Point2d dv;
    auto problem = test::createLinefitProblem(ets, dv, paramTrue, 1000, 0.1, 2.0);
    auto p0 = dv._v;
    const double initialError = test::evaluateError(*problem);

    // Now let's optimize.
    OptimizerRprop::Options options;
    options.maxIterations = 500;
    options.numThreadsGradient = 8;
    options.convergenceDeltaObjective = 0.0;
    options.convergenceDeltaX = 0.0;
    options.convergenceGradientNorm = 1e-6;
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    for (OptimizerRprop::Options::Method method : {OptimizerRprop::Options::RPROP_PLUS, OptimizerRprop::Options::RPROP_MINUS,
      OptimizerRprop::Options::IRPROP_MINUS, OptimizerRprop::Options::IRPROP_PLUS}) {

      optimizer.getOptions().method = method;
      optimizer.initialize();
      dv._v = p0; // reset design variable
      SCOPED_TRACE("");
      optimizer.optimize();
      auto ret = optimizer.getStatus();
      EXPECT_TRUE(ret.success());
      EXPECT_LE(ret.gradientNorm, options.convergenceGradientNorm);
      EXPECT_GE(ret.error, 0.0);
      EXPECT_LT(test::evaluateError(*problem), initialError);
      EXPECT_LT(ret.maxDeltaX, 1e-3);
      if (method == OptimizerRprop::Options::IRPROP_PLUS)
        EXPECT_LT(ret.deltaError, 1e-12);
      sm::eigen::assertNear(dv._v, paramTrue, 1e-1, SM_SOURCE_FILE_POS, "");
    }

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerRpropTestSuite, testRpropSquaredErrorTerms)
{
  try {
    using namespace aslam::backend;
    boost::shared_ptr<OptimizationProblem> problem_ptr(new OptimizationProblem);
    OptimizationProblem& problem = *problem_ptr;
    const int P = 2;
    const int E = 3;
    // Add some design variables.
    std::vector< boost::shared_ptr<Point2d> > p2d;
    p2d.reserve(P);
    for (int p = 0; p < P; ++p) {
      boost::shared_ptr<Point2d> point(new Point2d(Eigen::Vector2d::Random())); // random initialization of design variable
      p2d.push_back(point);
      problem.addDesignVariable(point);
      point->setBlockIndex(p);
      point->setActive(true);
    }

    // make a deep copy
    std::vector< boost::shared_ptr<Point2d> > p2d0;
    p2d0.reserve(p2d.size());
    for (auto& dv : p2d) p2d0.emplace_back(new Point2d(*dv));

    // Add some error terms.
    std::vector< boost::shared_ptr<LinearErr> > e1;
    e1.reserve(P*E);
    for (int p = 0; p < P; ++p) {
      for (int e = 0; e < E; ++e) {
        boost::shared_ptr<LinearErr> err(new LinearErr(p2d[p].get()));
        e1.push_back(err);
        problem.addErrorTerm(err);
        SCOPED_TRACE("");
        testErrorTerm(err);
      }
    }
    // Now let's optimize.
    OptimizerRprop::Options options;
    options.method = OptimizerRprop::Options::RPROP_PLUS;
    options.maxIterations = 500;
    options.numThreadsGradient = 8;
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem_ptr);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    for (OptimizerRprop::Options::Method method : {OptimizerRprop::Options::RPROP_PLUS, OptimizerRprop::Options::RPROP_MINUS,
      OptimizerRprop::Options::IRPROP_MINUS, OptimizerRprop::Options::IRPROP_PLUS}) {

      optimizer.getOptions().method = method;
      optimizer.initialize();
      for (std::size_t i=0; i<p2d.size(); i++) p2d[i]->_v = p2d0[i]->_v;
      SCOPED_TRACE("");
      optimizer.optimize();
      auto ret = optimizer.getStatus();
      EXPECT_TRUE(ret.success());
      EXPECT_LT(ret.gradientNorm, 1e-3);
      EXPECT_GE(ret.error, 0.0);
      EXPECT_LT(ret.maxDeltaX, 1e-3);
      if (method == OptimizerRprop::Options::IRPROP_PLUS)
        EXPECT_LT(ret.deltaError, 1e-12);
    }

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

