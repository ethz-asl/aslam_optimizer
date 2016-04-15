#include <sm/eigen/gtest.hpp>
#include <aslam/backend/OptimizerMultistart.hpp>
#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <sm/random.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

TEST(OptimizerTestSuite, testMultistart)
{
  try
  {
    using namespace aslam::backend;
    boost::shared_ptr<OptimizationProblem> problem_ptr(new OptimizationProblem);
    OptimizationProblem& problem = *problem_ptr;

    // Add a design variables.
    Scalar dv(6.0); // initialize to converge into basin of attraction with higher cost
    problem.addDesignVariable(&dv, false);
    dv.setBlockIndex(0);
    dv.setColumnBase(0);
    dv.setActive(true);

    // Construct a bimodal error function as a sum of two basins of attraction
    const double mu = 3., sigma = 2.;
    NegatedSquaredExponentialError1d e1(&dv, -mu, sigma, 1.0), e2(&dv, mu, sigma, 0.8);
    problem.addErrorTerm(&e1, false);
    problem.addErrorTerm(&e2, false);
    testErrorTerm(e1);
    testErrorTerm(e2);

    // Now let's optimize.
    OptimizerOptionsBFGS options;
    options.convergenceGradientNorm = 1e-10;
    options.linesearch.maxStepLength = 200;
    auto initFunctor = std::bind(randomRestarts, std::placeholders::_1, 20 /* nRestarts*/, [] (int) { return sm::random::randLU(-7.0, 7.0); });
    OptimizerMultistart optimizer(boost::shared_ptr<OptimizerBFGS>(new OptimizerBFGS(options)), initFunctor);
    optimizer.setProblem(problem_ptr);
    EXPECT_NO_THROW(optimizer.checkProblemSetup());
    optimizer.initialize();

    optimizer.optimize();
    auto ret = optimizer.getStatus();
    EXPECT_TRUE(ret.success());
    EXPECT_LE(ret.gradientNorm, options.convergenceGradientNorm); // make sure the optimizer did converge with sufficient accuracy
    EXPECT_LT(ret.error, -0.999); // make sure we found the basin with lower costs
    EXPECT_NEAR(dv._v(0,0), -mu, 1e-3);
  }
  catch (const std::exception& e)
  {
    FAIL() << e.what();
  }
}
