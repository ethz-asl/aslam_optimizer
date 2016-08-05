/*
 * LeapFrog.cpp
 *
 *  Created on: 4.08.2016
 *      Author: Ulrich Schwesinger
 */


#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/util/ProblemManager.hpp>
#include <aslam/backend/LeapFrog.hpp>

#include <aslam/Exceptions.hpp>

using namespace boost::python;
using namespace aslam::backend;

tuple leapFrogWrapper0(ProblemManager& pm,
                       aslam::backend::ColumnVectorType momentum,
                       const std::size_t numSteps,
                       const double stepLength,
                       const std::size_t numThreadsGradient = 4) {

  SM_ASSERT_EQ(aslam::InvalidArgumentException, static_cast<std::size_t>(momentum.size()), pm.numOptParameters(),
               "Invalid size of momentum vector");
  const auto potEnergy = getCostFunction(pm, false, true, false, numThreadsGradient, 1);
  RowVectorType gradient;
  potEnergy->computeGradient(gradient);
  const auto valid = aslam::backend::leapfrog::simulate(potEnergy, gradient, momentum, numSteps, stepLength);
  return make_tuple(valid, gradient, momentum);
}

tuple leapFrogWrapper1(const boost::shared_ptr<OptimizationProblem>& problem,
                       aslam::backend::ColumnVectorType momentum,
                       const std::size_t numSteps,
                       const double stepLength,
                       const std::size_t numThreadsGradient = 4) {
  ProblemManager pm;
  pm.setProblem(problem);
  pm.initialize();
  return leapFrogWrapper0(pm, momentum, numSteps, stepLength, numThreadsGradient);
}


BOOST_PYTHON_FUNCTION_OVERLOADS(leapFrogWrapper0_overloads, leapFrogWrapper0, 4, 5);
BOOST_PYTHON_FUNCTION_OVERLOADS(leapFrogWrapper1_overloads, leapFrogWrapper1, 4, 5);

void exportLeapFrog()
{
  def("simulateLeapFrog", &leapFrogWrapper0, leapFrogWrapper0_overloads());
  def("simulateLeapFrog", &leapFrogWrapper1, leapFrogWrapper1_overloads());
}

