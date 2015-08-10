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

struct SamplerHmcOptions {
  SamplerHmcOptions();
  SamplerHmcOptions(const sm::PropertyTree& config);
  void check() const;

  double delta;    /// \brief
  size_t nHamiltonianSteps;
  size_t nThreads;
};

inline std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerHmcOptions& options)
{
  out << "SamplerHmcOptions:\n";
  out << "\tdelta: " << options.delta << std::endl;
  out << "\tnHamiltonianSteps: " << options.nHamiltonianSteps << std::endl;
  out << "\tnThreads: " << options.nThreads << std::endl;
  return out;
}

/**
 * @class SamplerHmc
 * @brief The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled.
 * It interprets the objective value of an optimization problem as the negative log density of a probability distribution.
 * The log density has to be defined up to proportionality of the true negative log density.
 */
class SamplerHmc {

 public:
  typedef SamplerHmc self_t;
  typedef boost::shared_ptr<self_t> Ptr;
  typedef boost::shared_ptr<const self_t> ConstPtr;
  typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorType;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorType;
  typedef boost::shared_ptr<OptimizationProblemBase> NegativeLogDensityPtr;
#ifdef aslam_backend_ENABLE_TIMING
  typedef sm::timing::Timer Timer;
#else
  typedef sm::timing::DummyTimer Timer;
#endif

  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

  struct Statistics {
    Statistics();
    void reset();
    /// \brief Getter for the acceptance rate
    double getAcceptanceRate(bool total = false) const { return getNumIterations(total) > 0 ? static_cast<double>(getNumAcceptedSamples(total))/static_cast<double>(getNumIterations(total)) : 0.0; }
    /// \brief Getter for the number of iterations since the last run() or initialize() call
    std::size_t getNumIterations(bool total = false) const { return total ? nIterationsTotal : nIterationsThisRun; }
    /// \brief Getter for the number of iterations since the last run() or initialize() call
    std::size_t getNumAcceptedSamples(bool total = false) const { return total ? nSamplesAcceptedTotal : nSamplesAcceptedThisRun; }

    std::size_t nIterationsThisRun; /// \brief How many iterations the sampler has run in the last run() call
    std::size_t nIterationsTotal; /// \brief How many iterations the sampler has run in the last initialize() call
    std::size_t nSamplesAcceptedThisRun; /// \brief How many samples were accepted since the last run() call
    std::size_t nSamplesAcceptedTotal; /// \brief How many samples were accepted since the last initialize() call
  };

 public:
  /// \brief Default constructor with default options
  SamplerHmc();
  /// \brief Constructor
  SamplerHmc(const SamplerHmcOptions& options);
  /// \brief Destructor
  ~SamplerHmc() { }

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
  SamplerHmcOptions& options() { return _options; }
  /// \brief Const getter for statistics
  Statistics& statistics() { return _statistics; }
  /// \brief Mutable getter for the log density formulation
  NegativeLogDensityPtr getNegativeLogDensity() { return _negLogDensity; }

 private:
  /// \brief Update the design variables
  void updateDesignVariables(const ColumnVectorType& dx);
  /// \brief Save the current state of the design variables
  void saveDesignVariables();
  /// \brief Revert to the last state saved by a call to saveDesignVariables()
  void revertUpdateDesignVariables();

  /// \brief compute the current gradient of the objective function
  void computeGradient(RowVectorType& outGrad, size_t nThreads, bool useMEstimator);
  void evaluateGradients(size_t threadId, size_t startIdx, size_t endIdx, bool useMEstimator, RowVectorType& grad);
  void setupThreadedJob(boost::function<void(size_t, size_t, size_t, bool, RowVectorType&)> job,
                        size_t nThreads, std::vector<RowVectorType>& out, bool useMEstimator);

 private:
  SamplerHmcOptions _options; /// \brief Configuration options
  NegativeLogDensityPtr _negLogDensity; /// \brief The negative log probability density

  /// \brief all design variables, first the non-marginalized ones (the dense ones), then the marginalized ones.
  std::vector<DesignVariable*> _designVariables;

  /// \brief all of the squared error terms involved in the log density
  std::vector<ErrorTerm*> _errorTermsS;
  /// \brief all of the non-squared error terms involved in the log density
  std::vector<ScalarNonSquaredErrorTerm*> _errorTermsNS;

  std::vector< std::pair<DesignVariable*, Eigen::MatrixXd> > _dvState;

  /// \brief the total number of parameters of this problem, given by number of design variables and their dimensionality
  std::size_t _numOptParameters;

  /// \brief the total number of error terms as the sum of squared and non-squared error terms
  std::size_t _numErrorTerms;

  bool _isInitialized; /// \brief Whether the optimizer is correctly initialized

  Statistics _statistics; /// \brief Statistics collected during run

};

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERMCMC_HPP_ */
