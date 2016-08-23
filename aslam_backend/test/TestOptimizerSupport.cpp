/*
 * TestOptimizerSupport.cpp
 *
 *  Created on: 25.05.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#include "TestOptimizerSupport.hpp"

#include <sm/random.hpp>

namespace aslam {
namespace backend {
namespace test {

boost::shared_ptr<OptimizationProblem> createLinefitProblem(std::vector< boost::shared_ptr<TestNonSquaredError> >& errorTerms,
                                                            Point2d& dv,
                                                            const Eigen::Vector2d& paramTrue,
                                                            const std::size_t numErrors /*= 1000*/,
                                                            const double outputNoiseStd /*= 0.1*/,
                                                            const double parameterNoiseStd /*= 10.0*/) {

  using namespace aslam::backend;

  boost::shared_ptr<OptimizationProblem> problem(new OptimizationProblem);

  // implement least squares line fit through data
  // Set up problem representing a linear function fit of the form y = a + b*x + N(0, outputNoiseStd)
  // to noisy samples from a linear function
  dv = Point2d(paramTrue + parameterNoiseStd*Eigen::Vector2d::NullaryExpr([&] (int) { return sm::random::randn(); })); // random initialization of design variable
  dv.setBlockIndex(0);
  dv.setActive(true);
  problem->addDesignVariable(&dv, false);

  errorTerms.clear();
  errorTerms.reserve(numErrors);
  for (std::size_t e = 0; e < numErrors; ++e) {
    const double x = static_cast<double>(e)/numErrors;
    const double y = paramTrue[0] + paramTrue[1]*x + outputNoiseStd*sm::random::randn();
    errorTerms.emplace_back(new TestNonSquaredError(&dv, x, y));
    problem->addErrorTerm(errorTerms.back());
  }
  return problem;
}

double evaluateError(const OptimizationProblemBase& problem)  {
  double error = 0.0;
  for (std::size_t i=0; i<problem.numErrorTerms(); ++i) error += const_cast<ErrorTerm*>(problem.errorTerm(i))->evaluateError();
  for (std::size_t i=0; i<problem.numNonSquaredErrorTerms(); ++i) error += const_cast<ScalarNonSquaredErrorTerm*>(problem.nonSquaredErrorTerm(i))->evaluateError();
  return error;
};

} /* namespace aslam */
} /* namespace backend */
} /* namespace test */
