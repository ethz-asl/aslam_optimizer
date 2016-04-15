/*
 * OptimizerMultistart.cpp
 *
 *  Created on: 11.04.2016
 *      Author: sculrich
 */

// self
#include <aslam/backend/OptimizerMultistart.hpp>

// Schweizer Messer
#include <sm/logging.hpp>

namespace aslam
{
namespace backend
{

OptimizerMultistart::OptimizerMultistart(boost::shared_ptr<OptimizerBase> optimizer, std::function<bool (const std::vector<DesignVariable*>&)> newStartFcn)
    : _optimizer(optimizer), _newStartFcn(newStartFcn)
{
  SM_ASSERT_FALSE(Exception, optimizer == nullptr, "You have to provide a valid optimizer");
}

void OptimizerStatusMultistart::resetImplementation()
{
  *this = OptimizerStatusMultistart();
}

void OptimizerMultistart::optimizeImplementation()
{
  static Eigen::IOFormat fmt(15, 0, ", ", ", ", "", "", "[", "]");

  this->reset();

  while (_newStartFcn(_optimizer->getDesignVariables())) // get new initialization point
  {
    SM_DEBUG_STREAM_NAMED("optimization", "OptimizerMultistart: Generated new initialization point " <<
                          utils::getFlattenedDesignVariableParameters(_optimizer->getDesignVariables()).transpose().format(fmt));
    _optimizer->reset();
    _optimizer->optimize(); // run until convergence
    _status.statuses.push_back(_optimizer->getStatus());
    _status.solutions.emplace_back(_optimizer->getDesignVariables());
    _status.numIterations += _optimizer->getStatus().numIterations;
    _status.numObjectiveEvaluations += _optimizer->getStatus().numObjectiveEvaluations;
    _status.numDerivativeEvaluations += _optimizer->getStatus().numDerivativeEvaluations;

    // update status
    if (_optimizer->getStatus().error < _status.error)
    {
      SM_DEBUG_STREAM_NAMED("optimization", "OptimizerMultistart: Improved solution " << _status.error << " -> " << _optimizer->getStatus().error);
      _status.bestSolutionIndex = _status.solutions.size() - 1;
      const auto numIterations = _status.numIterations;
      const auto numObjectiveEvaluations = _status.numObjectiveEvaluations;
      const auto numDerivativeEvaluations = _status.numDerivativeEvaluations;
      static_cast<OptimizerStatus&>(_status) = _optimizer->getStatus();
      _status.numIterations = numIterations;
      _status.numObjectiveEvaluations = numObjectiveEvaluations;
      _status.numDerivativeEvaluations = numDerivativeEvaluations;
    }
  }

  if (!_status.solutions.empty())
  {
    _status.solutions.at(_status.bestSolutionIndex).restore();
    SM_DEBUG_STREAM_NAMED("optimization", "OptimizerMultistart: No new initialization points available, "
        "terminating with best solution " << utils::getFlattenedDesignVariableParameters(_optimizer->getDesignVariables()).transpose().format(fmt) << std::endl <<
        static_cast<OptimizerStatus&>(_status));
  }
  else
  {
    SM_ERROR_STREAM("OptimizerMultistart: No valid solution could be found among " << _status.solutions.size() << " solutions.");
  }
}

} /* namespace aslam */
} /* namespace backend */
