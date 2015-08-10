/*
 * SamplerMcmc.hpp
 *
 *  Created on: 24.07.2015
 *      Author: sculrich
 */

#ifndef INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_
#define INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_

#include <limits>

#include <sm/BoostPropertyTree.hpp>

#include "SamplerBase.hpp"
//#include "OptimizationProblemBase.hpp"
//#include "DesignVariable.hpp"
//#include "ErrorTerm.hpp"
//#include "ScalarNonSquaredErrorTerm.hpp"

namespace aslam {
namespace backend {

struct SamplerMetropolisHastingsOptions {
  SamplerMetropolisHastingsOptions();
  SamplerMetropolisHastingsOptions(const sm::PropertyTree& config);
  double transitionKernelSigma;  /// \brief Standard deviation for the Gaussian Markov transition kernel \f$ \\mathcal{N(\mathbf 0, \text{diag{\sigma^2})} f$
};

inline std::ostream& operator<<(std::ostream& out, const aslam::backend::SamplerMetropolisHastingsOptions& options)
{
  out << "SamplerMetropolisHastingsOptions:\n";
  out << "\ttransitionKernelSigma: " << options.transitionKernelSigma << std::endl;
  return out;
}

/**
 * @class SamplerMetropolisHastings
 * @brief The sampler returns samples (design variables) of a probability distribution that cannot be directly sampled.
 * It interprets the objective value of an optimization problem as the negative log density of a probability distribution.
 * The log density has to be defined up to proportionality of the true negative log density.
 */
class SamplerMetropolisHastings : public SamplerBase {

 public:
  typedef boost::shared_ptr<SamplerMetropolisHastings> Ptr;
  typedef boost::shared_ptr<const SamplerMetropolisHastings> ConstPtr;

  SM_DEFINE_EXCEPTION(Exception, aslam::Exception);

 public:
  /// \brief Default constructor with default options
  SamplerMetropolisHastings();
  /// \brief Constructor
  SamplerMetropolisHastings(const SamplerMetropolisHastingsOptions& options);
  /// \brief Destructor
  ~SamplerMetropolisHastings() { }

  /// \brief Mutable getter for options
  SamplerMetropolisHastingsOptions& options() { return _options; }

  /// \brief Getter for the acceptance rate
  double getAcceptanceRate() const { return _nIterations > 0 ? static_cast<double>(_nSamplesAccepted)/static_cast<double>(_nIterations) : 0.0; }

  /// \brief Getter for the number of iterations since the last run() or initialize() call
  std::size_t getNumIterations() const { return _nIterations; }

 private:
  virtual void initializeImplementation();
  virtual void runImplementation(const std::size_t nStepsMax, const std::size_t nAcceptedSamples);

 private:
   SamplerMetropolisHastingsOptions _options; /// \brief Configuration options

  std::size_t _nIterations; /// \brief How many iterations the sampler has run
  std::size_t _nSamplesAccepted; /// \brief How many samples were accepted since the last run() or initialize() call

};

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_SAMPLERMETROPOLISHASTINGS_HPP_ */
