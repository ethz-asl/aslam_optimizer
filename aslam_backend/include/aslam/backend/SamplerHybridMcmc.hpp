/*
 * SamplerHmc.hpp
 *
 * Hamiltonian Markov-Chain Monte Carlo Sampler
 *
 *  Created on: Aug 05, 2015
 *      Author: sculrich
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERHYBRIDMCMC_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERHYBRIDMCMC_HPP_

#include <sm/BoostPropertyTree.hpp>

#include "SamplerBase.hpp"

namespace sm {
  class PropertyTree;
}

namespace aslam {
namespace backend {

struct SamplerHybridMcmcOptions {
  SamplerHybridMcmcOptions();
  SamplerHybridMcmcOptions(const sm::PropertyTree& config);
  void check() const;

  double leapFrogStepSize; /// \brief Step length for the Leap-Frog integration
  size_t nLeapFrogSteps; /// \brief Number of steps for the Leap-Frog integration
  size_t nThreads; /// \brief Number of threads to use for gradient computation
};

std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerHybridMcmcOptions& options);

/**
 * @class SamplerHybridMcmc
 * @brief Hybrid Markov Chain Monte-Carlo Sampler using Hamiltonian dynamics.
 *
 * The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled.
 * It interprets the objective value of an optimization problem as the negative log density of a probability distribution.
 * The log density has to be defined up to proportionality of the true negative log density.
 */
class SamplerHybridMcmc : public SamplerBase {

 public:
  typedef boost::shared_ptr<SamplerHybridMcmc> Ptr;
  typedef boost::shared_ptr<const SamplerHybridMcmc> ConstPtr;

 public:
  /// \brief Default constructor with default options
  SamplerHybridMcmc();
  /// \brief Constructor
  SamplerHybridMcmc(const SamplerHybridMcmcOptions& options);
  /// \brief Destructor
  ~SamplerHybridMcmc() { }

  /// \brief Initialization method
  virtual void initialize() override;

  /// \brief Mutable getter for options
  SamplerHybridMcmcOptions& options() { return _options; }

 private:

  /// \brief Implementation of the step method
  virtual void step(bool& accepted, double& acceptanceProbability);

  /// \brief Save the current state of the design variables
  void saveDesignVariables();
  /// \brief Revert to the last state saved by a call to saveDesignVariables()
  void revertUpdateDesignVariables();

 private:
  SamplerHybridMcmcOptions _options; /// \brief Configuration options
  std::vector< std::pair<DesignVariable*, Eigen::MatrixXd> > _dvState; /// \brief State of a set of design variables

  RowVectorType _gradient; /// \brief Current gradient of the negative log density
  double _u; /// \brief Current potential energy of the system
  bool _lastSampleAccepted; /// \brief Whether the last sample was accepted or not

};

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERMCMC_HPP_ */