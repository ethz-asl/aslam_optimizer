/*
 * OptimizerMultistart.hpp
 *
 *  Created on: 11.04.2016
 *      Author: sculrich
 */

#ifndef INCLUDE_ASLAM_BACKEND_OPTIMIZERMULTISTART_HPP_
#define INCLUDE_ASLAM_BACKEND_OPTIMIZERMULTISTART_HPP_

// standard
#include <functional>

// Eigen
#include <Eigen/Dense>

// Schweizer Messer
#include <sm/logging.hpp>

// self
#include <aslam/Exceptions.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/OptimizerBase.hpp>
#include <aslam/backend/util/utils.hpp> // DesignVariableState

namespace aslam
{
namespace backend
{

struct OptimizerStatusMultistart : public OptimizerStatus
{
  int bestSolutionIndex = -1;
  std::vector<OptimizerStatus> statuses;
  std::vector<utils::DesignVariableState> solutions;
 private:
  void resetImplementation() override;
};

/**
 * \class OptimizerMultistart
 * Run optimization from multiple initialization points
 */
class OptimizerMultistart : public OptimizerBase
{
 public:
  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

  /**
   * Constructor
   * @param optimizer The underlying optimizer to use
   * @param newStartFcn Functor to initialize design variables.
   *                    Return false if no new initialization point available.
   */
  OptimizerMultistart(boost::shared_ptr<OptimizerBase> optimizer, std::function<bool (const std::vector<DesignVariable*>&)> newStartFcn);

  void setProblem(boost::shared_ptr<OptimizationProblemBase> problem) override { _optimizer->setProblem(problem); }
  void checkProblemSetup() override { _optimizer->checkProblemSetup(); }
  bool isInitialized() { return _optimizer->isInitialized(); }
  const std::vector<DesignVariable*>& getDesignVariables() { return _optimizer->getDesignVariables(); }
  const OptimizerStatus& getStatus() const override { return _status; }
  const OptimizerOptionsBase& getOptions() const override { return _optimizer->getOptions(); }
  void setOptions(const OptimizerOptionsBase& options) override { _optimizer->setOptions(options); }

 private:
  void optimizeImplementation() override;
  void resetImplementation() override { _optimizer->reset(); }
  void initializeImplementation() override { _optimizer->initialize(); }

 private:
  boost::shared_ptr<OptimizerBase> _optimizer; /// \brief Underlying optimizer to use
  std::function<bool (const std::vector<DesignVariable*>&)> _newStartFcn; /// \brief New initialization point functor
  OptimizerStatusMultistart _status; /// \brief Status of the multistart optimizer
}; /* class OptimizerMultistart */

bool randomRestarts(const std::vector<DesignVariable*>& dvs, const std::size_t nRestarts, std::function<double(int)> random)
{
  static std::size_t cnt = 0;
  if (cnt++ >= nRestarts)
    return false;

  for (auto dv : dvs)
    dv->setParameters(Eigen::MatrixXd::NullaryExpr(dv->minimalDimensions(), 1, random));
  SM_DEBUG_STREAM_NAMED("optimization", "RandomRestart: sampled new initialization point " << cnt << "/" << nRestarts << ": " <<
                        utils::getFlattenedDesignVariableParameters(dvs).transpose());
  return true;
}

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_OPTIMIZERMULTISTART_HPP_ */
