/*
 * OptimizerCallback.cpp
 *
 *  Created on: 24.08.2016
 *      Author: Ulrich Schwesinger
 */

#include <string>
#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/OptimizerBase.hpp>
#include <aslam/backend/OptimizerCallback.hpp>
#include <aslam/backend/OptimizerCallbackManager.hpp>
#include <aslam/python/ExportOptimizerCallbackEvent.hpp>

using namespace boost::python;
using namespace aslam::python;
using namespace aslam::backend::callback;

std::type_index event2typeid(const boost::python::object& o)
{
  boost::python::extract<std::type_index> ext(boost::python::getattr(o, "__typeId", boost::python::object()));
  if(ext.check()) { return ext(); }
  throw std::runtime_error("Invalid event type!");
}

void addCallbackWrapper(Registry& registry, const boost::python::object& event, const boost::python::object& callback)
{
  if (!boost::python::getattr(callback, "__call__", boost::python::object())) {
    throw std::runtime_error("Invalid callback object, has to be a callable!");
  }
  if (boost::python::getattr(event, "__getitem__", boost::python::object())) {
    for (int i=0; i<len(event); ++i) {
      registry.add( {event2typeid(event[i])} , [callback]() {
        callback();
      });
    }
  } else {
    registry.add(event2typeid(event), [callback]() {
      callback();
    });
  }
}


void exportOptimizerCallback()
{
  class_<std::type_index>("type_info", no_init);

  class_<Event>("__CallbackEvent")
    .def_readonly("currentCost", &Event::currentCost)
    .def_readonly("previousLowestCost", &Event::previousLowestCost)
  ;

  exportEvent<event::OPTIMIZATION_INITIALIZED>("EVENT_OPTIMIZATION_INITIALIZED",
                                               "Right after the the initial cost has been computed.");
  exportEvent<event::ITERATION_START>("EVENT_ITERATION_START",
                                      "At the start of an iteration before any work has been done, "
                                      "same state as in PER_ITERATION_END in previous iteration");
  exportEvent<event::ITERATION_END>("EVENT_ITERATION_END",
                                    "At the end of an iteration after all work has been done, "
                                    "same state as in PER_ITERATION_START in next iteration");
  exportEvent<event::LINEAR_SYSTEM_SOLVED>("EVENT_LINEAR_SYSTEM_SOLVED",
                                           "Right after the linear system was solved");
  exportEvent<event::DESIGN_VARIABLES_UPDATED>("EVENT_DESIGN_VARIABLES_UPDATED",
                                               "Right after the design variables (X) have been updated.");
  exportEvent<event::RESIDUALS_UPDATED>("EVENT_RESIDUALS_UPDATED",
                                        "After the raw squared error for each error term has been updated but "
                                        "before the m-estimators are applied to compute the effective cost. "
                                        "This is the right event to update m-estimators based on residuals.");
  exportEvent<event::COST_UPDATED>("EVENT_COST_UPDATED",
                                   "Right after the m-estimators are applied to compute the effective cost.");

  class_<Registry>("CallbackRegistry")
    .def("add", &addCallbackWrapper, "Adds a callback for a specific event or a list of events")

    .def("clear", (void (Registry::*)(void))&Registry::clear, "Removes all callbacks")

    .def("clear", detail::make_function_aux(
        [](Registry& registry, const object& event)
        { registry.clear(event2typeid(event)); },
        default_call_policies(), boost::mpl::vector<void, Registry&, const object&>()),
         "Removes all callbacks for a specific event")

    .def("numCallbacks", detail::make_function_aux(
        [](const Registry& registry, const object& event)
        { return registry.numCallbacks(event2typeid(event)); },
        default_call_policies(), boost::mpl::vector<std::size_t, const Registry&, const object&>()),
         "Number of callbacks for a specific event")
  ;
}

