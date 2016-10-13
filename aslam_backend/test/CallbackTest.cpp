#include <sm/eigen/gtest.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <aslam/backend/test/ErrorTermTestHarness.hpp>
#include "SampleDvAndError.hpp"


TEST(CallbackTestSuite, testCallback)
{
  try {
    using namespace aslam::backend;
    boost::shared_ptr<OptimizationProblem> problem_ptr(new OptimizationProblem);
    OptimizationProblem& problem = *problem_ptr;
    const int P = 3;
    const int E = 2;
    // Add some design variables.
    std::vector< boost::shared_ptr<Point2d> > p2d;
    for (int p = 0; p < P; ++p) {
      boost::shared_ptr<Point2d> point(new Point2d(Eigen::Vector2d::Random()));
      p2d.push_back(point);
      problem.addDesignVariable(point);
      point->setBlockIndex(p);
      point->setActive(true);
    }
    // Add some error terms.
    std::vector< boost::shared_ptr<LinearErr> > errorTerms;
    for (int p = 0; p < P; ++p) {
      for (int e = 0; e < E; ++e) {
        boost::shared_ptr<LinearErr> err(new LinearErr(p2d[p].get()));
        errorTerms.push_back(err);
        problem.addErrorTerm(err);
      }
    }

    // Now let's optimize.
    OptimizerOptions options;
    options.verbose = false;
    options.linearSolver = "dense";
    options.levenbergMarquardtLambdaInit = 10;
    options.doSchurComplement = false;
    options.doLevenbergMarquardt = false;
    options.maxIterations = 3;
    Optimizer optimizer(options);
    optimizer.setProblem(problem_ptr);

    using namespace callback;
    int countInit = 0, countResUpdate = 0, countCostUpdate = 0, countDesignUpdate = 0;
    double startJ, lastUpdatedJ, expectedCost;
    optimizer.callback().add<event::OPTIMIZATION_INITIALIZED>(
        [&](const Event & arg) {
          countInit ++;
          lastUpdatedJ = startJ = arg.currentCost;
          ASSERT_EQ(typeid(event::OPTIMIZATION_INITIALIZED), typeid(arg));
        }
      );
    optimizer.callback().add<event::DESIGN_VARIABLES_UPDATED>(
        [&](const Event & arg) {
          countDesignUpdate ++;
          ASSERT_EQ(typeid(event::DESIGN_VARIABLES_UPDATED), typeid(arg));
        }
      );
    optimizer.callback().add<event::RESIDUALS_UPDATED>(
        [&](const event::RESIDUALS_UPDATED & arg) {
          countResUpdate ++;
          expectedCost = 0;
          for(auto && e : errorTerms){
            expectedCost += e->getSquaredError();
          }
          ASSERT_EQ(typeid(event::RESIDUALS_UPDATED), typeid(arg));
        }
      );
    optimizer.callback().add<event::COST_UPDATED>(
        [&](const event::COST_UPDATED & arg) { // void with argument
          countCostUpdate ++;
          ASSERT_DOUBLE_EQ(expectedCost, arg.currentCost);
          lastUpdatedJ = arg.currentCost;
          ASSERT_EQ(typeid(event::COST_UPDATED), typeid(arg));
        }
      );
    optimizer.callback().add({typeid(event::COST_UPDATED), typeid(event::COST_UPDATED)}, // register twice
        [&](const Event &) { // void with argument
          countCostUpdate ++;
        }
      );
    optimizer.callback().add<event::COST_UPDATED>(
        [&](const Event &) { // ProceedInstruction with argument
          countCostUpdate ++;
          return ProceedInstruction::CONTINUE;
        }
      );
    optimizer.callback().add<event::COST_UPDATED>(
        [&]() { // ProceedInstruction without argument
          countCostUpdate ++;
          return ProceedInstruction::CONTINUE;
        }
      );

    auto ret = optimizer.optimize();

    ASSERT_EQ(1, countInit);
    EXPECT_EQ(ret.iterations + 1, countResUpdate);
    EXPECT_EQ((ret.iterations + 1) * 5, countCostUpdate);
    EXPECT_EQ(ret.iterations, countDesignUpdate);
    EXPECT_EQ(ret.JStart, startJ);
    EXPECT_EQ(ret.JFinal, lastUpdatedJ);
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}



