/*
 * ExportOptimizerCallbackEvent.hpp
 *
 *  Created on: 24.08.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#ifndef INCLUDE_ASLAM_PYTHON_EXPORTOPTIMIZERCALLBACKEVENT_HPP_
#define INCLUDE_ASLAM_PYTHON_EXPORTOPTIMIZERCALLBACKEVENT_HPP_

#include <string>
#include <typeindex>
#include <boost/python.hpp>
#include <aslam/backend/OptimizerCallback.hpp>

namespace aslam {
namespace python {

namespace details {

template <typename T, typename... Rest>
void addTypeInfoAttribute(boost::python::class_<T, Rest...>&& bpclass)
{
  bpclass.attr("__typeId") = std::type_index{typeid(T)};
}

} /* namespace details */

template <typename EVENT>
void exportEvent(const std::string& name, const std::string& desc = std::string()) {
  details::addTypeInfoAttribute(
      boost::python::class_<EVENT, boost::python::bases<aslam::backend::callback::Event>>(name.c_str(), desc.c_str(), boost::python::no_init)
  );
}

} /* namespace python */
} /* namespace aslam */

#endif /* INCLUDE_ASLAM_PYTHON_EXPORTOPTIMIZERCALLBACKEVENT_HPP_ */
