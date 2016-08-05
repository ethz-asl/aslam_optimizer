/*
 * ProblemManager.cpp
 *
 *  Created on: 05.08.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#include <numpy_eigen/boost_python_headers.hpp>

#include <aslam/backend/util/ProblemManager.hpp>
#include <aslam/backend/OptimizationProblem.hpp>

using namespace boost::python;
using namespace aslam::backend;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(evaluateError_overloads, evaluateError, 0, 1);

void exportProblemManager()
{
  class_< ProblemManager, boost::shared_ptr<ProblemManager> >("ProblemManager", init<>())
    .def(init< boost::shared_ptr<OptimizationProblemBase> >())
    .def("setProblem", &ProblemManager::setProblem)
    .def("initialize", &ProblemManager::initialize)
    .add_property("isInitialized", &ProblemManager::isInitialized)
    .add_property("problem", (boost::shared_ptr<const OptimizationProblemBase> (ProblemManager::*) (void) const)&ProblemManager::getProblem)
    .def("designVariable", make_function(&ProblemManager::designVariable, return_internal_reference<>()))
    .add_property("numDesignVariables", &ProblemManager::numDesignVariables)
    .add_property("designVariables", make_function(&ProblemManager::designVariables, return_value_policy<copy_const_reference>()))
    .add_property("numOptParameters", &ProblemManager::numOptParameters)
    .add_property("numErrorTerms", &ProblemManager::numErrorTerms)
    .def("checkProblemSetup", &ProblemManager::checkProblemSetup)
    .def("evaluateError", &ProblemManager::evaluateError, evaluateError_overloads())
    .def("signalProblemChanged", &ProblemManager::signalProblemChanged)
    .def("applyStateUpdate", &ProblemManager::applyStateUpdate)
    .def("revertLastStateUpdate", &ProblemManager::revertLastStateUpdate)
    .def("saveDesignVariables", &ProblemManager::saveDesignVariables)
    .def("restoreDesignVariables", &ProblemManager::restoreDesignVariables)
    .def("getFlattenedDesignVariableParameters", &ProblemManager::getFlattenedDesignVariableParameters)
    .def("computeGradient", &ProblemManager::computeGradient)
    .def("applyDesignVariableScaling", &ProblemManager::applyDesignVariableScaling)
  ;
}
