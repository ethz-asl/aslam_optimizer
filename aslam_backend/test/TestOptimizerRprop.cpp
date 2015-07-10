#include <sm/eigen/gtest.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <sm/random.hpp>
#include <aslam/backend/test/ErrorTermTestHarness.hpp>
#include "SampleDvAndError.hpp"


TEST(OptimizerRpropTestSuite, testRpropNonSquaredErrorTerms)
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
    // Add some error terms.
    std::vector< boost::shared_ptr<TestNonSquaredError> > e1;
    e1.reserve(P*E);
    for (int p = 0; p < P; ++p) {
      for (int e = 0; e < E; ++e) {
        TestNonSquaredError::grad_t g(p+1, e+1);
        boost::shared_ptr<TestNonSquaredError> err(new TestNonSquaredError(p2d[p].get(), g));
        err->_p = 1.0;
        e1.push_back(err);
        problem.addErrorTerm(err);
      }
    }
    // Now let's optimize.
    OptimizerRpropOptions options;
    options.verbose = false;
    options.maxIterations = 500;
    options.nThreads = 1;
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem_ptr);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    optimizer.optimize();

    EXPECT_LT(optimizer.getGradientNorm(), 1e-3);

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
    // Add some error terms.
    std::vector< boost::shared_ptr<LinearErr> > e1;
    e1.reserve(P*E);
    for (int p = 0; p < P; ++p) {
      for (int e = 0; e < E; ++e) {
        boost::shared_ptr<LinearErr> err(new LinearErr(p2d[p].get()));
        e1.push_back(err);
        problem.addErrorTerm(err);
      }
    }
    // Now let's optimize.
    OptimizerRpropOptions options;
    options.verbose = false;
    options.maxIterations = 500;
    options.nThreads = 1;
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem_ptr);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    optimizer.optimize();

    EXPECT_LT(optimizer.getGradientNorm(), 1e-3);
//    EXPECT_NEAR(p2d[0]->_v[0], 1.0, 1e-3);

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

