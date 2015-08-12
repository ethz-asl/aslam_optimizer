/*
 * SamplerBase.hpp
 *
 *  Created on: 10.08.2015
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_

#include <limits>

#include <aslam/backend/util/ProblemManager.hpp>

namespace aslam {
namespace backend {

class SamplerBase {

 public:
  class Statistics {
   public:
    friend class SamplerBase;

    Statistics();
    ~Statistics() { }

    /// \brief Reset all information, i.e. counters to zero
    void reset();
    /// \brief Getter for the acceptance rate
    double getAcceptanceRate(bool total = false) const;
    /// \brief Getter for the number of iterations since the last run() or initialize() call
    std::size_t getNumIterations(bool total = false) const;
    /// \brief Getter for the number of iterations since the last run() or initialize() call
    std::size_t getNumAcceptedSamples(bool total = false) const;

   public:
    std::size_t nIterationsThisRun; /// \brief How many iterations the sampler has run in the last run() call
    std::size_t nSamplesAcceptedThisRun; /// \brief How many samples were accepted since the last run() call

   private:
    std::size_t nIterationsTotal; /// \brief How many iterations the sampler has run in the last initialize() call
    std::size_t nSamplesAcceptedTotal; /// \brief How many samples were accepted since the last initialize() call
  };

 public:
  virtual ~SamplerBase() { }

  /// \brief Initialization method
  virtual void initialize();

  /// \brief Run the sampler for at maximum \p nStepsMax until \p nAcceptedSamples samples were accepted
  void run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples = std::numeric_limits<std::size_t>::max());

  /// \brief Set up to work on the log density. The log density may neglect the normalization constant.
  void setNegativeLogDensity(boost::shared_ptr<OptimizationProblemBase> negLogDensity);

  /// \brief Mutable getter for the log density formulation
  boost::shared_ptr<OptimizationProblemBase> getNegativeLogDensity();

  /// \brief Const getter for the log density formulation
  boost::shared_ptr<const OptimizationProblemBase> getNegativeLogDensity() const;

  /// \brief Signal the sampler that the negative log density formulation changed.
  void signalNegativeLogDensityChanged();

  /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
  ///        hooked up to design variables and running finite differences on error terms where this is possible.
  void checkNegativeLogDensitySetup() const;

  /// \brief Const getter for statistics
  const Statistics& statistics() const;

 protected:
  /// \brief Evaluate the current negative log density
  double evaluateNegativeLogDensity() const;

  ProblemManager& getProblemManager() { return _problemManager; }
 private:
  /// \brief Run the sampler for at maximum \p nStepsMax until \p nAcceptedSamples samples were accepted
  virtual void runImplementation(const std::size_t nStepsMax, const std::size_t nAcceptedSamples, Statistics& statistics) = 0;

 private:
  Statistics _statistics; /// \brief statistics collected during runtime
  ProblemManager _problemManager;  /// \brief the manager for the attached problem
};

}
}

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERBASE_HPP_ */
