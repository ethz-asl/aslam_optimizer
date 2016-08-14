#include <sm/eigen/gtest.hpp>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <initializer_list>

#include <sm/random.hpp>
#include <sm/BoostPropertyTree.hpp>

#include <aslam/backend/OptimizerSgd.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>

#include "SampleDvAndError.hpp"
#include "TestOptimizerSupport.hpp"

TEST(OptimizerStochasticGradientDescentTestSuite, testLearningRateScheduleConstant)
{
  try {
    using namespace aslam::backend;
    {
      LearningRateScheduleConstant schedule(2.0);
      EXPECT_DOUBLE_EQ(-2.0, schedule.step(0, 0, RowVectorType::Ones(1,1))[0]);
      EXPECT_DOUBLE_EQ(-2.0, schedule.step(0, 1, RowVectorType::Ones(1,1))[0]);
    }
    {
      sm::BoostPropertyTree pt;
      pt.setDouble("constant", -2.0);
      EXPECT_ANY_THROW(LearningRateScheduleConstant schedule(pt));
      pt.setDouble("constant", 2.0);
      LearningRateScheduleConstant schedule(pt);
      EXPECT_DOUBLE_EQ(pt.getDouble("constant"), schedule.rate());
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerStochasticGradientDescentTestSuite, testLearningRateScheduleOptimal)
{
  try {
    using namespace aslam::backend;
    {
      LearningRateScheduleOptimal schedule(2.0, 0.5);
      EXPECT_DOUBLE_EQ(-2.0, schedule.step(0, 0, RowVectorType::Ones(1,1))[0]);
      EXPECT_DOUBLE_EQ(-2.*2.0/(1.0 + 2.0*1.0/0.5), schedule.step(0, 1, 2.*RowVectorType::Ones(1,1))[0]);
    }
    {
      sm::BoostPropertyTree pt;
      pt.setDouble("initialRate", 2.0);
      pt.setDouble("tau", -2.0);
      EXPECT_ANY_THROW(LearningRateScheduleOptimal schedule(pt));
      pt.setDouble("tau", 2.0);
      LearningRateScheduleOptimal schedule(pt);
      EXPECT_DOUBLE_EQ(pt.getDouble("tau"), schedule.tau());
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerStochasticGradientDescentTestSuite, testLearningRateScheduleAdaDelta)
{
  try {
    using namespace aslam::backend;
    {
      LearningRateScheduleAdaDelta schedule(0.95, 1e-3);
//      EXPECT_DOUBLE_EQ(2.0, schedule.step(0, 0, RowVectorType::Ones(1,1))[0]);
//      EXPECT_DOUBLE_EQ(2.0/(1.0 + 2.0*1.0/0.5), schedule.step(0, 1, RowVectorType::Ones(1,1))[0]);
    }
    {
      sm::BoostPropertyTree pt;
      pt.setDouble("decayRate", 0.95);
      pt.setDouble("eps", -1.);
      EXPECT_ANY_THROW(LearningRateScheduleAdaDelta schedule(pt));
      pt.setDouble("eps", 1e-3);
      LearningRateScheduleAdaDelta schedule(pt);
      EXPECT_DOUBLE_EQ(pt.getDouble("decayRate"), schedule.decayRate());
      EXPECT_DOUBLE_EQ(pt.getDouble("eps"), schedule.eps());
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerStochasticGradientDescentTestSuite, testOptions)
{
  try {
    using namespace aslam::backend;
    OptimizerSgd::Options options;
    TEST_INVALID_OPTION(options, learningRateSchedule, NULL);
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(OptimizerStochasticGradientDescentTestSuite, testOptimizerStochasticGradientDescent)
{
  try {

    using namespace aslam::backend;

    sm::random::seed(std::time(NULL));
    std::srand ( unsigned ( std::time(NULL) ) ); // for random_shuffle

    const Eigen::Vector2d paramTrue(1.0, 2.0);
    std::vector< boost::shared_ptr<TestNonSquaredError> > ets;
    Point2d dv;
    auto problem = test::createLinefitProblem(ets, dv, paramTrue, 1000, 0.1);
    const double initialError = test::evaluateError(*problem);

    // Now let's optimize.
    OptimizerSgd::Options options;
    options.useDenseJacobianContainer = true;
    options.learningRateSchedule.reset(new LearningRateScheduleOptimal(1.0, ets.size()/2.0));
    options.batchSize = 1;
    options.maxIterations = 1;
    OptimizerSgd optimizer(options);
    EXPECT_EQ(0, optimizer.getStatus().numIterations);

    std::size_t cnt = 0;
    const std::size_t numPasses = 1;
    for (size_t i=0; i<numPasses; i++) {
      std::random_shuffle(ets.begin(), ets.end());
      for (auto& et : ets) {
        SCOPED_TRACE("Adding batch");
        optimizer.addBatch< std::initializer_list<boost::shared_ptr<TestNonSquaredError> > >( { et } );
        SCOPED_TRACE("Optimizing");
        optimizer.optimize();
        SM_VERBOSE_STREAM("f(x) = " << dv._v[0] << " + " << dv._v[1] << " * x, error = " << test::evaluateError(*problem));
        cnt++;
        ASSERT_EQ(cnt, optimizer.getStatus().numIterations);
        ASSERT_EQ(cnt, optimizer.getStatus().numSubIterations);
        ASSERT_EQ(i*ets.size() + cnt, optimizer.getStatus().numTotalIterations);
      }
    }

    optimizer.setCounters(numPasses+1, cnt+2);
    EXPECT_EQ(numPasses+1, optimizer.getStatus().numIterations);
    EXPECT_EQ(cnt+2, optimizer.getStatus().numTotalIterations);

    const auto& ret = optimizer.getStatus();
    EXPECT_EQ(numPasses*ets.size(), ret.numDerivativeEvaluations);
    EXPECT_GE(ret.maxDeltaX, 0.0);
    EXPECT_LT(test::evaluateError(*problem), initialError);
    sm::eigen::assertNear(dv._v, paramTrue, 1e-1, SM_SOURCE_FILE_POS, "");

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
