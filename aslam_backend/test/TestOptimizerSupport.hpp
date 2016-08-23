/*
 * TestOptimizerSupport.hpp
 *
 *  Created on: 25.05.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#ifndef TEST_TESTOPTIMIZERSUPPORT_HPP_
#define TEST_TESTOPTIMIZERSUPPORT_HPP_

#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

#include <aslam/backend/OptimizationProblem.hpp>
#include "SampleDvAndError.hpp"

namespace aslam {
namespace backend {
namespace test {

/**
 * Creates a least-squares linefit optimization problem
 * @param[out] errorTerms Error terms in the optimization problem
 * @param[out] dv The single design variable of the problem
 * @param[in] paramTrue True parameters of the line, i.e. y = f(x) = paramTrue[0] + paramTrue[1]*x
 * @param[in] numErrors Number of error terms to be added
 * @param[in] outputNoiseStd Standard deviation of noise on output values
 * @param[in] parameterNoiseStd Standard deviation of noise on initial values for design variable \p dv
 * @return Filled optimization problem
 */
boost::shared_ptr<OptimizationProblem> createLinefitProblem(std::vector< boost::shared_ptr<TestNonSquaredError> >& errorTerms,
                                                            Point2d& dv,
                                                            const Eigen::Vector2d& paramTrue,
                                                            const std::size_t numErrors = 1000,
                                                            const double outputNoiseStd = 0.1,
                                                            const double parameterNoiseStd = 10.0);

/// @brief Evaluate the error of the optimization problem \p problem
double evaluateError(const OptimizationProblemBase& problem);

#define TEST_INVALID_OPTION(options, field, invalidValue) { \
    const auto val = options.field; \
    options.field = invalidValue; \
    EXPECT_ANY_THROW(options.check()) << "Invalid value " << invalidValue << " for option " << #field << " not detected"; \
    options.field = val; \
  }

} /* namespace aslam */
} /* namespace backend */
} /* namespace test */

#endif /* TEST_TESTOPTIMIZERSUPPORT_HPP_ */
