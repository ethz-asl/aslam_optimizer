/*
 * SamplerMcmc.hpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERMCMC_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERMCMC_HPP_

#include <boost/shared_ptr.hpp>

#include <sm/timing/Timer.hpp>
#include <sm/BoostPropertyTree.hpp>

#include "OptimizationProblemBase.hpp"
#include "DesignVariable.hpp"
#include "ErrorTerm.hpp"
#include "ScalarNonSquaredErrorTerm.hpp"

namespace aslam {
namespace backend {

struct SamplerMcmcOptions {
  SamplerMcmcOptions() ;
  SamplerMcmcOptions(const sm::PropertyTree& config);

  std::size_t nSamples; /// \brief How many samples to produce
  std::size_t nStepsBurnIn; /// \brief Number of Markov steps to perform in convergence phase
  std::size_t nStepsSkip; /// \brief Number of Markov steps to skip in between samples returned
  double transitionKernelSigma;  /// \brief Standard deviation for the Gaussian Markov transition kernel \f$ \\mathcal{N(\mathbf 0, \text{diag{\sigma^2})} f$
};

inline std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerMcmcOptions& options)
{
  out << "SamplerMcmcOptions:\n";
  out << "\tnSamples: " << options.nSamples << std::endl;
  out << "\tnStepsBurnIn: " << options.nStepsBurnIn << std::endl;
  out << "\tnStepsSkip: " << options.nStepsSkip << std::endl;
  out << "\ttransitionKernelSigma: " << options.transitionKernelSigma << std::endl;
  return out;
}

/**
 * @class SamplerMcmc
 * @brief The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled.
 * It interprets the objective value of an optimization problem as the log density of a probability distribution.
 * The log density has to be defined up to proportionality of the true log density.
 */
class SamplerMcmc {

 public:
  typedef SamplerMcmc self_t;
  typedef boost::shared_ptr<self_t> Ptr;
  typedef boost::shared_ptr<const self_t> ConstPtr;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
  typedef boost::shared_ptr<OptimizationProblemBase> LogDensityPtr;
#ifdef aslam_backend_ENABLE_TIMING
  typedef sm::timing::Timer Timer;
#else
  typedef sm::timing::DummyTimer Timer;
#endif

  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

 public:
  /// \brief Constructor
  SamplerMcmc(const SamplerMcmcOptions& options);
  /// \brief Destructor
  ~SamplerMcmc() { }

  /// \brief Set up to work on the log density. The log density may neglect the normalization constant.
  void setLogDensity(LogDensityPtr logDensity);
  /// \brief initialize the sampler
  void initialize();
  /// \brief Signal the sampler that the log density formulation changed.
  void signalLogDensityChanged() { _isInitialized = false; }
  /// \brief Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is
  ///        hooked up to design variables and running finite differences on error terms where this is possible.
  void checkLogDensitySetup();

  /// \brief Run the sampler to returns samples from the distribution.
  Eigen::MatrixXd run();

 private:

  /// \brief Return the design variables of the log density as vector
  void getDesignVariables(ColumnVectorType& x) const;
  /// \brief Update the design variables from a vector
  void updateDesignVariables(const ColumnVectorType& dx);
  /// \brief Revert the last update performed by \ref updateDesignVariables
  void revertUpdateDesignVariables();
  /// \brief Compute the diagonal entries of the transition kernel represented as an uncorrelated normal distribution
  void evaluateTransitionKernel(ColumnVectorType& dx);
  /// \brief Evaluate the log density
  double computeLogDensity() const;

 private:
  SamplerMcmcOptions _options;  /// \brief Configuration options
  LogDensityPtr _problem; /// \brief The log probability density

  /// \brief all design variables, first the non-marginalized ones (the dense ones), then the marginalized ones.
  std::vector<DesignVariable*> _designVariables;
  /// \brief the total number of parameters, given by number of design variables and their dimensionality
  std::size_t _numParameters;

  /// \brief all of the squared error terms involved in the log density
  std::vector<ErrorTerm*> _errorTermsS;
  /// \brief all of the non-squared error terms involved in the log density
  std::vector<ScalarNonSquaredErrorTerm*> _errorTermsNS;

  bool _isInitialized;          /// \brief Whether the optimizer is correctly initialized
  std::size_t _nIterations;     /// \brief How many iterations the sampler has run

};

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERMCMC_HPP_ */
