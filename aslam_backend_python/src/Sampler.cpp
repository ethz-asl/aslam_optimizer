/*
 * SamplerBase.cpp
 *
 *  Created on: 12.08.2015
 *      Author: Ulrich Schwesinger
 */


#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/SamplerBase.hpp>
#include <aslam/backend/SamplerMetropolisHastings.hpp>
#include <aslam/backend/OptimizationProblemBase.hpp>

using namespace boost::python;
using namespace aslam::backend;

template <typename T>
std::string toString(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

void exportSampler()
{

  class_<SamplerBase::Statistics, boost::noncopyable>("SamplerStatistics", init<>("Statistics(): Default constructor"))
      .def("reset", &SamplerBase::Statistics::reset)
      .def("getAcceptanceRate", &SamplerBase::Statistics::getAcceptanceRate)
      .def("getNumIterations", &SamplerBase::Statistics::getNumIterations)
      .def("getNumAcceptedSamples", &SamplerBase::Statistics::getNumAcceptedSamples)
      .def("getWeightedMeanAcceptanceProbability", &SamplerBase::Statistics::getWeightedMeanAcceptanceProbability)
      .def("updateWeightedMeanAcceptanceProbability", &SamplerBase::Statistics::updateWeightedMeanAcceptanceProbability)
  ;

  class_<SamplerBase, boost::shared_ptr<SamplerBase> , boost::noncopyable>("SamplerBase", no_init)
      .def("initialize", &SamplerBase::initialize)
      .def("run", &SamplerBase::run)
      .def("reset", &SamplerBase::reset)
      .def("setNegativeLogDensity", &SamplerBase::setNegativeLogDensity)
      .def("getNegativeLogDensity", (boost::shared_ptr<const OptimizationProblemBase> (SamplerBase::*) (void) const)&SamplerBase::getNegativeLogDensity)
      .def("signalNegativeLogDensityChanged", &SamplerBase::signalNegativeLogDensityChanged)
      .def("checkNegativeLogDensitySetup", &SamplerBase::checkNegativeLogDensitySetup)
      .add_property("statistics", make_function(&SamplerBase::statistics, return_internal_reference<>()))
  ;
  implicitly_convertible< boost::shared_ptr<SamplerBase>, boost::shared_ptr<const SamplerBase> >();


  class_<SamplerMetropolisHastingsOptions>("SamplerMetropolisHastingsOptions",
                                           "Options for the Metropolis-Hastings sampler",
                                           init<>("SamplerMetropolisHastingsOptions(): Default constructor"))

      .def_readwrite("transitionKernelSigma", &SamplerMetropolisHastingsOptions::transitionKernelSigma,
                     "Standard deviation for the Gaussian Markov transition kernel N(0, transitionKernelSigma^2)")
      .def_readwrite("nThreadsEvaluateLogDensity", &SamplerMetropolisHastingsOptions::nThreadsEvaluateLogDensity,
                     "How many threads to use to evaluate the error terms involved in the negative log density")
      .def("__str__", &toString<SamplerMetropolisHastingsOptions>)
  ;


  std::string classDocString = "The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled."
      " It interprets the objective value of an optimization problem as the negative log density of a probability distribution."
      " The log density has to be defined up to proportionality of the true negative log density.";

  class_<SamplerMetropolisHastings, boost::shared_ptr<SamplerMetropolisHastings>, bases<SamplerBase> >("SamplerMetropolisHastings",
                                                                                                       classDocString.c_str(),
                                                                                                       no_init)

      .def(init<>("SamplerMetropolisHastings(): Default constructor"))
      .def(init<const SamplerMetropolisHastingsOptions&>("SamplerMetropolisHastings(SamplerMetropolisHastingsOptions options): Constructor with custom options"))

      .add_property("options", make_function(&SamplerMetropolisHastings::options, return_internal_reference<>()), "Mutable getter for options")

    ;
  implicitly_convertible< boost::shared_ptr<SamplerMetropolisHastings>, boost::shared_ptr<const SamplerMetropolisHastings> >();

}

