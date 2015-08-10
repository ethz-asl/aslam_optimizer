/*
 * SamplerMcmc.hpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_

#include <limits>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <sm/timing/Timer.hpp>
#include <sm/BoostPropertyTree.hpp>

#include "OptimizationProblemBase.hpp"
#include "DesignVariable.hpp"
#include "ErrorTerm.hpp"
#include "ScalarNonSquaredErrorTerm.hpp"

namespace aslam {
namespace backend {

struct SamplerMetropolisHastingsOptions {
  SamplerMetropolisHastingsOptions();
  SamplerMetropolisHastingsOptions(const sm::PropertyTree& config);
  double transitionKernelSigma;  /// \brief Standard deviation for the Gaussian Markov transition kernel \f$ \\mathcal{N(\mathbf 0, \text{diag{\sigma^2})} f$
};

inline std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerMetropolisHastingsOptions& options)
{
  out << "SamplerMcmcOptions:\n";
  out << "\ttransitionKernelSigma: " << options.transitionKernelSigma << std::endl;
  return out;
}

/**
 * @class SamplerMcmc
 * @brief The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled.
 * It interprets the objective value of an optimization problem as the negative log density of a probability distribution.
 * The log density has to be defined up to proportionality of the true negative log density.
 */
class SamplerMetropolisHastings {

 public:
  typedef SamplerMetropolisHastings self_t;
  typedef boost::shared_ptr<self_t> Ptr;
  typedef boost::shared_ptr<const self_t> ConstPtr;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
  typedef boost::shared_ptr<OptimizationProblemBase> NegativeLogDensityPtr;
#ifdef aslam_backend_ENABLE_TIMING
  typedef sm::timing::Timer Timer;
#else
  typedef sm::timing::DummyTimer Timer;
#endif

  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

 public:
  /// \brief Default constructor with default options
  SamplerMetropolisHastings();
  /// \brief Constructor
  SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options);
  /// \brief Destructor
  ~SamplerMetropolisHastings() { }

  /// \brief Set up to work on the log density. The log density may neglect the normalization constant.
  void setNegativeLogDensity(NegativeLogDensityPtr negLogDensity);
  /// \brief initialize the sampler
  void initialize();
  /// \brief Signal the sampler that the negative log density formulation changed.
  void signalNegativeLogDensityChanged() { _isInitialized = false; }
  /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
  ///        hooked up to design variables and running finite differences on error terms where this is possible.
  void checkNegativeLogDensitySetup();

  /// \brief Run the sampler for nSteps.
  ///        If a burn-in phase is desired, call run() with the desired number of burn-in steps before using the state of the design variables.
  ///        In order to get uncorrelated samples, call run() with nSteps >> 1 and use the state of the design variables after nSteps.
  /// \param nSteps Run for this number of steps at maximum
  /// \param nAcceptedSamples Run until this number of samples was accepted
  void run(const std::size_t nStepsMax, const std::size_t nAcceptedSamples = std::numeric_limits<std::size_t>::max());

  /// \brief Evaluate the current log density
  double evaluateNegativeLogDensity() const;

  /// \brief Mutable getter for options
  SamplerMetropolisHastingsOptions& options() { return _options; }
  /// \brief Mutable getter for the log density formulation
  NegativeLogDensityPtr getNegativeLogDensity() { return _negLogDensity; }
  /// \brief Getter for the acceptance rate
  double getAcceptanceRate() const { return _nIterations > 0 ? static_cast<double>(_nSamplesAccepted)/static_cast<double>(_nIterations) : 0.0; }
  /// \brief Getter for the number of iterations since the last run() or initialize() call
  std::size_t getNumIterations() const { return _nIterations; }

 private:
  /// \brief Update the design variables based on the Gaussian transition kernel
  void updateDesignVariables();
  /// \brief Revert the last update performed by \ref updateDesignVariables()
  void revertUpdateDesignVariables();

 private:
  SamplerMetropolisHastingsOptions _options; /// \brief Configuration options
  NegativeLogDensityPtr _negLogDensity; /// \brief The negative log probability density

  /// \brief all design variables, first the non-marginalized ones (the dense ones), then the marginalized ones.
  std::vector<DesignVariable*> _designVariables;

  /// \brief all of the squared error terms involved in the log density
  std::vector<ErrorTerm*> _errorTermsS;
  /// \brief all of the non-squared error terms involved in the log density
  std::vector<ScalarNonSquaredErrorTerm*> _errorTermsNS;

  bool _isInitialized; /// \brief Whether the optimizer is correctly initialized
  std::size_t _nIterations; /// \brief How many iterations the sampler has run
  std::size_t _nSamplesAccepted; /// \brief How many samples were accepted since the last run() or initialize() call

};

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_ */
