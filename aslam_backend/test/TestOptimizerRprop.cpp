#include <sm/eigen/gtest.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <sm/random.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
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

    // make a deep copy
    std::vector< boost::shared_ptr<Point2d> > p2d0;
    p2d0.reserve(p2d.size());
    for (auto& dv : p2d) p2d0.emplace_back(new Point2d(*dv));

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
        SCOPED_TRACE("");
        testErrorTerm(err);
      }
    }
    // Now let's optimize.
    OptimizerRpropOptions options;
    options.maxIterations = 500;
    options.nThreads = 8;
    options.convergenceGradientNorm = 0.0;
    options.convergenceDx = 0.0;
    EXPECT_ANY_THROW(options.check());
    options.convergenceGradientNorm = 1e-6;
    EXPECT_NO_THROW(options.check());
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem_ptr);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    for (OptimizerRpropOptions::Method method : {OptimizerRpropOptions::RPROP_PLUS, OptimizerRpropOptions::RPROP_MINUS,
      OptimizerRpropOptions::IRPROP_MINUS, OptimizerRpropOptions::IRPROP_PLUS}) {

      options.method = method;
      optimizer.options() = options;
      optimizer.initialize();
      for (std::size_t i=0; i<p2d.size(); i++) p2d[i]->_v = p2d0[i]->_v;
      SCOPED_TRACE("");
      auto ret = optimizer.optimize();
      EXPECT_TRUE(ret.success());
      EXPECT_LE(optimizer.getGradientNorm(), 1e-6);
      EXPECT_GE(ret.error, 0.0);
      EXPECT_LT(ret.maxDx, 1e-3);
      if (method == OptimizerRpropOptions::IRPROP_PLUS)
        EXPECT_LT(ret.derror, 1e-12);
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
    OptimizerRpropOptions options;
    options.method = OptimizerRpropOptions::RPROP_PLUS;
    options.maxIterations = 500;
    options.nThreads = 8;
    OptimizerRprop optimizer(options);
    optimizer.setProblem(problem_ptr);

    EXPECT_NO_THROW(optimizer.checkProblemSetup());

    for (OptimizerRpropOptions::Method method : {OptimizerRpropOptions::RPROP_PLUS, OptimizerRpropOptions::RPROP_MINUS,
      OptimizerRpropOptions::IRPROP_MINUS, OptimizerRpropOptions::IRPROP_PLUS}) {

      options.method = method;
      optimizer.options() = options;
      optimizer.initialize();
      for (std::size_t i=0; i<p2d.size(); i++) p2d[i]->_v = p2d0[i]->_v;
      SCOPED_TRACE("");
      auto ret = optimizer.optimize();
      EXPECT_TRUE(ret.success());
      EXPECT_LT(optimizer.getGradientNorm(), 1e-3);
      EXPECT_GE(ret.error, 0.0);
      EXPECT_LT(ret.maxDx, 1e-3);
      if (method == OptimizerRpropOptions::IRPROP_PLUS)
        EXPECT_LT(ret.derror, 1e-12);
    }

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

