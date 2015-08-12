/*
 * ProblemManager.cpp
 *
 *  Created on: 12.08.2015
 *      Author: Ulrich Schwesinger
 */

#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/OptimizationProblemBase.hpp>
#include <aslam/backend/util/ProblemManager.hpp>

using namespace boost::python;
using namespace aslam::backend;

void exportProblemManager()
{

  class_<ProblemManager, boost::shared_ptr<ProblemManager> >("ProblemManager", init<>("ProblemManager(): Constructor with default options"))
    .def("setProblem", &ProblemManager::setProblem, "Set up to work on the optimization problem.")
    .def("initialize", &ProblemManager::initialize, "Initialize the class")
    .def("isInitialized", &ProblemManager::isInitialized, "Is everything initialized?")
    .def("getProblem", (boost::shared_ptr<const OptimizationProblemBase> (ProblemManager::*) (void) const)&ProblemManager::getProblem, "Getter for the optimization problem")
    .def("designVariable", make_function(&ProblemManager::designVariable, return_internal_reference<>()), "Get dense design variable i.")
    .def("numDesignVariables", &ProblemManager::numDesignVariables, "how many dense design variables are involved in the problem")
    .def("numOptParameters", &ProblemManager::numOptParameters, "how many scalar parameters (design variables with their minimal dimension) are involved in the problem")
    .def("numErrorTerms", &ProblemManager::numErrorTerms, "how many error terms are involved in the problem")
    .def("checkProblemSetup", &ProblemManager::checkProblemSetup,
         "Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is hooked up to design variables and running "
         "finite differences on error terms where this is possible.")
    .def("evaluateError", &ProblemManager::evaluateError, "Evaluate the value of the objective function")
    .def("signalProblemChanged", &ProblemManager::signalProblemChanged, "Signal that the problem changed.")
    .def("applyStateUpdate", &ProblemManager::applyStateUpdate, "Apply the update vector to the design variables")
    .def("revertLastStateUpdate", &ProblemManager::revertLastStateUpdate, "Undo the last state update to the design variables")
  ;

  implicitly_convertible< boost::shared_ptr<ProblemManager>, boost::shared_ptr<const ProblemManager> >();

}
