/*
 * SamplerBase.hpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_

#include <limits>

#include <aslam/backend/ScalarOptimizerBase.hpp>

namespace aslam {
namespace backend {

class SamplerBase : public ScalarOptimizerBase {

 public:
  SamplerBase() : ScalarOptimizerBase() { }
  virtual ~SamplerBase() { }

  /// \brief Run the sampler for at maximum \ref nStepsMax until \ref nAcceptedSamples samples were accepted
  void run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples = std::numeric_limits<std::size_t>::max()) {

    SM_ASSERT_GT(Exception, nStepsMax, 0, "It does not make sense to run the sampler with no steps.");
    SM_ASSERT_GT(Exception, nAcceptedSamples, 0, "It does not make sense to run the sampler until zero samples were accepted.");

    if (!isInitialized())
      initialize();

    runImplementation(nStepsMax, nAcceptedSamples);
  }

  /// \brief Set up to work on the log density. The log density may neglect the normalization constant.
  void setNegativeLogDensity(boost::shared_ptr<OptimizationProblemBase> negLogDensity) { setProblem(negLogDensity); }

  /// \brief Mutable getter for the log density formulation
  boost::shared_ptr<OptimizationProblemBase> getNegativeLogDensity() { return getProblem(); }

  /// \brief Signal the sampler that the negative log density formulation changed.
  void signalNegativeLogDensityChanged() { setInitialized(false); }

  /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
  ///        hooked up to design variables and running finite differences on error terms where this is possible.
  void checkNegativeLogDensitySetup() { checkProblemSetup(); }

 protected:
  /// \brief Evaluate the current negative log density
  double evaluateNegativeLogDensity() const { return evaluateError(); }

 private:
  /// \brief Run the sampler for at maximum \ref nStepsMax until \ref nAcceptedSamples samples were accepted
  virtual void runImplementation(const std::size_t nStepsMax, const std::size_t nAcceptedSamples) = 0;

};

}
}

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_ */
