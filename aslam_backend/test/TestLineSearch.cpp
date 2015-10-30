#include <sm/eigen/gtest.hpp>
#include <aslam/backend/LineSearch.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

#include <sm/random.hpp>

using namespace std;
using namespace aslam::backend;

class ErrorTermLS : public ScalarNonSquaredErrorTerm {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Scalar* _s;                      /// \brief The scalar design variable
  ErrorTermLS(Scalar* s) : _s(s) {
    setDesignVariables(_s);
  }
  virtual ~ErrorTermLS() {}
  virtual double evaluateErrorImplementation() {
    return _s->_v[0]*_s->_v[0];
  }
  virtual void evaluateJacobiansImplementation(aslam::backend::JacobianContainer & outJ) {
    outJ.add(_s, 2.*_s->_v);
  }
};

void checkWolfeConditions(double error0, double error1, double derror0, double derror1, double c1, double c2, double stepLength) {
  EXPECT_LE(error1, error0 + c1*stepLength*derror0);
  EXPECT_GE(derror1, c2*derror0);
}

TEST(LineSearchTestSuite, testLineSearch)
{
  try {
    using namespace aslam::backend;

    const double x = sm::random::randLU(-10.0, 10.0); // initial abscissa
    Scalar dv( (Scalar::Vector1d() << x).finished() );
    dv.setActive(true);

    ErrorTermLS errorTerm(&dv);
    errorTerm.setWeight(1.0);
    SCOPED_TRACE("");
    testErrorTerm(errorTerm);

    boost::shared_ptr<OptimizationProblem> problem_ptr(new OptimizationProblem);
    OptimizationProblem& problem = *problem_ptr;
    problem.addDesignVariable(&dv, false);
    problem.addErrorTerm(&errorTerm, false);

    ProblemManager pm;
    pm.setProblem(problem_ptr);
    pm.checkProblemSetup();
    pm.initialize();
    ASSERT_EQ(1, pm.numOptParameters());

    LineSearch ls(&pm);

    // Compute initial error
    const double error0 = pm.evaluateError(1);

    // Compute search direction
    RowVectorType grad0 = RowVectorType::Zero(1);
    pm.computeGradient(grad0, 1, true);
    RowVectorType searchDirection = -grad0;

    {
      // Test initialization
      EXPECT_ANY_THROW(ls.getError()); // not initialized yet
      EXPECT_ANY_THROW(ls.getErrorDerivative()); // not initialized yet

      ls.initialize(searchDirection);
      EXPECT_DOUBLE_EQ(error0, ls.getError());
      EXPECT_TRUE(ls.getGradient().isApprox(grad0));

      ls.initialize(searchDirection, 0.0);
      EXPECT_DOUBLE_EQ(0.0, ls.getError());
      EXPECT_TRUE(ls.getGradient().isApprox(grad0));

      RowVectorType testGrad = RowVectorType::Random(1);
      ls.initialize(searchDirection, 0.0, testGrad);
      EXPECT_DOUBLE_EQ(0.0, ls.getError());
      EXPECT_TRUE(ls.getGradient().isApprox(testGrad));

      ls.initialize(searchDirection, boost::optional<double>(), testGrad);
      EXPECT_DOUBLE_EQ(error0, ls.getError());
      EXPECT_TRUE(ls.getGradient().isApprox(testGrad));
    }

    const double derror0 = ls.computeErrorDerivative();

    enum LineSearchMethod { WOLFE1, WOLFE2, WOLFE12 };

    for ( auto& method : {WOLFE1, WOLFE2, WOLFE12} ) {

      dv._v[0] = x;
      ls.initialize(searchDirection);
      SCOPED_TRACE("");

      bool success;
      switch (method) {
        case WOLFE1:
          success = ls.lineSearchWolfe1();
          break;
        case WOLFE2:
          success = ls.lineSearchWolfe2();
          break;
        case WOLFE12:
          success = ls.lineSearchWolfe12();
          break;
      }
      ASSERT_TRUE(success);

      // Check strong Wolfe conditions
      const double error1 = ls.getError();
      const double derror1 = ls.computeErrorDerivative();
      RowVectorType grad1;
      pm.computeGradient(grad1, 1, true);
      SCOPED_TRACE("");
      checkWolfeConditions(error0, error1, derror0, derror1, ls.options().c1WolfeCondition,
                           ls.options().c2WolfeCondition, ls.getCurrentStepLength());
    }

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
