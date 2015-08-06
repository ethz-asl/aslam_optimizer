#include <sm/eigen/gtest.hpp>
#include <sm/random.hpp>
#include <aslam/backend/SamplerMcmc.hpp>
#include <aslam/backend/SamplerHmc.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

using namespace std;
using namespace boost;
using namespace aslam::backend;

/// \brief Encodes the error \f$ -\frac{\left(\mathbf x - \mathbf \mu\right)^2}{2.0 \sigma^2}\f$
class GaussianNegLogDensityError : public ScalarNonSquaredErrorTerm {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef ScalarNonSquaredErrorTerm parent_t;

  Scalar* _x; /// \brief The design variable
  Scalar::Vector1d _mu;  /// \brief The mean
  Scalar::Vector1d _varInv;  /// \brief The inverse variance

  GaussianNegLogDensityError(Scalar* x) : _x(x) {
    _x->setActive(true);
    parent_t::setDesignVariables(_x);
    setWeight(1.0);
    _mu << 0.0;
    _varInv << 1.0;
  }
  virtual ~GaussianNegLogDensityError() {}

  void setMean(const double mu) { _mu << mu; }
  void setVariance(const double var) { _varInv << 1./var; }

  /// \brief evaluate the error term
  virtual double evaluateErrorImplementation() {
    const Scalar::Vector1d dv = (_x->_v - _mu);
    return 0.5*(_varInv*dv*dv)[0];
  }

  /// \brief evaluate the jacobian
  virtual void evaluateJacobiansImplementation(aslam::backend::JacobianContainer & outJ) {
    outJ.add( _x, (_x->_v - _mu)*_varInv );
  }

};

boost::shared_ptr<OptimizationProblem> setupProblem(const double meanTrue, const double sigmaTrue) {

  sm::random::seed(std::time(nullptr));

  boost::shared_ptr<OptimizationProblem> gaussian1dLogDensityPtr(new OptimizationProblem);
  OptimizationProblem& gaussian1dLogDensity = *gaussian1dLogDensityPtr;

  Scalar::Vector1d x;
  x << meanTrue + 10.0*sm::random::randn();

  // Add some design variables.
  boost::shared_ptr<Scalar> sdv(new Scalar(x));
  gaussian1dLogDensity.addDesignVariable(sdv);
  sdv->setBlockIndex(0);
  sdv->setActive(true);
  // Add error term
  boost::shared_ptr<GaussianNegLogDensityError> err(new GaussianNegLogDensityError(sdv.get()));
  err->setMean(meanTrue);
  err->setVariance(sigmaTrue*sigmaTrue);
  gaussian1dLogDensity.addErrorTerm(err);
  SCOPED_TRACE("");
  testErrorTerm(err);

  return gaussian1dLogDensityPtr;
}


TEST(OptimizerSamplerMcmcTestSuite, testSamplerMcmc)
{
  try {

    sm::random::seed(std::time(nullptr));

    const double meanTrue = 10.0;
    const double sigmaTrue = 2.0;
    boost::shared_ptr<OptimizationProblem> gaussian1dLogDensityPtr = setupProblem(meanTrue, sigmaTrue);
    OptimizationProblem& gaussian1dLogDensity = *gaussian1dLogDensityPtr;

    // Initialize and test options
    sm::BoostPropertyTree pt;
    pt.setDouble("transitionKernelSigma", 1.0);
    SamplerMcmcOptions options(pt);
    EXPECT_DOUBLE_EQ(pt.getDouble("transitionKernelSigma"), options.transitionKernelSigma);

    // Set and test log density
    SamplerMcmc sampler(options);
    sampler.setNegativeLogDensity(gaussian1dLogDensityPtr);
    EXPECT_NO_THROW(sampler.checkNegativeLogDensitySetup());
    EXPECT_DOUBLE_EQ(sampler.getAcceptanceRate(), 0.0);
    EXPECT_EQ(sampler.getNumIterations(), 0);

    // Parameters
    const int nSamples = 1000;
    const int nStepsBurnIn = 100;
    const int nStepsSkip = 50;

    // Burn-in
    sampler.run(nStepsBurnIn);
    EXPECT_GE(sampler.getAcceptanceRate(), 1e-3);
    EXPECT_LE(sampler.getAcceptanceRate(), 1.0);
    EXPECT_EQ(sampler.getNumIterations(), nStepsBurnIn);

    // Now let's retrieve samples
    Eigen::VectorXd dvValues(nSamples);
    for (size_t i=0; i<nSamples; i++) {
      sampler.run(nStepsSkip);
      EXPECT_GE(sampler.getAcceptanceRate(), 0.0);
      EXPECT_LE(sampler.getAcceptanceRate(), 1.0);
      EXPECT_EQ(sampler.getNumIterations(), nStepsSkip);
      ASSERT_EQ(1, gaussian1dLogDensity.numDesignVariables());
      auto dv = gaussian1dLogDensity.designVariable(0);
      Eigen::MatrixXd p;
      dv->getParameters(p);
      ASSERT_EQ(1, p.size());
      dvValues[i] = p(0,0);
    }

    // check sample mean
    EXPECT_NEAR(dvValues.mean(), meanTrue, 4.*sigmaTrue) << "This failure does not necessarily have to be an error. It should just appear "
        " with a probability of 0.00633 %";

    // check sample variance
    EXPECT_NEAR((dvValues.array() - dvValues.mean()).matrix().squaredNorm()/(dvValues.rows() - 1.0), sigmaTrue*sigmaTrue, 1e0) << "This failure does "
        "not necessarily have to be an error. It should just appear very rarely";

    // Run until a specified number of samples was accepted
    sampler.run(numeric_limits<size_t>::max(), 1);
    EXPECT_GT(sampler.getAcceptanceRate(), 0.0);
    EXPECT_GT(sampler.getNumIterations(), 0);

    // Check that re-initializing resets values
    sampler.initialize();
    EXPECT_DOUBLE_EQ(sampler.getAcceptanceRate(), 0.0);
    EXPECT_EQ(sampler.getNumIterations(), 0);

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}


#include <sm/logging.hpp>

TEST(OptimizerSamplerMcmcTestSuite, testSamplerHmc)
{
  try {

    sm::random::seed(std::time(nullptr));

    const double meanTrue = 10.0;
    const double sigmaTrue = 2.0;
    boost::shared_ptr<OptimizationProblem> gaussian1dLogDensityPtr = setupProblem(meanTrue, sigmaTrue);
    OptimizationProblem& gaussian1dLogDensity = *gaussian1dLogDensityPtr;

    // Initialize and test options
    sm::BoostPropertyTree pt;
    pt.setDouble("delta", 0.1);
    pt.setDouble("nHamiltonianSteps", 50);
    pt.setDouble("nThreads", 1);
    SamplerHmcOptions options(pt);
    EXPECT_DOUBLE_EQ(pt.getDouble("delta"), options.delta);
    EXPECT_DOUBLE_EQ(pt.getInt("nHamiltonianSteps"), options.nHamiltonianSteps);
    EXPECT_DOUBLE_EQ(pt.getInt("nThreads"), options.nThreads);

    // Set and test log density
    SamplerHmc sampler(options);
    sampler.setNegativeLogDensity(gaussian1dLogDensityPtr);
    EXPECT_NO_THROW(sampler.checkNegativeLogDensitySetup());
    EXPECT_DOUBLE_EQ(sampler.statistics().getAcceptanceRate(), 0.0);
    EXPECT_EQ(sampler.statistics().getNumIterations(), 0);

    // Parameters
    const int nSamples = 1000;
    const int nStepsBurnIn = 10;
    const int nStepsSkip = 5;

    // Burn-in
    sampler.run(nStepsBurnIn);
    EXPECT_GT(sampler.statistics().getAcceptanceRate(false), 0.0);
    EXPECT_LE(sampler.statistics().getAcceptanceRate(false), 1.0);
    EXPECT_GT(sampler.statistics().getAcceptanceRate(true), 0.0);
    EXPECT_LE(sampler.statistics().getAcceptanceRate(true), 1.0);
    EXPECT_EQ(sampler.statistics().getNumIterations(true), nStepsBurnIn);
    EXPECT_EQ(sampler.statistics().getNumIterations(false), nStepsBurnIn);

    // Now let's retrieve samples
    Eigen::VectorXd dvValues(nSamples);
    for (size_t i=0; i<nSamples; i++) {
      sampler.run(nStepsSkip);
      EXPECT_GE(sampler.statistics().getAcceptanceRate(), 0.0);
      EXPECT_LE(sampler.statistics().getAcceptanceRate(), 1.0);
      EXPECT_EQ(sampler.statistics().getNumIterations(false), nStepsSkip);
      EXPECT_EQ(sampler.statistics().getNumIterations(true), (i+1)*nStepsSkip + nStepsBurnIn);
      ASSERT_EQ(1, gaussian1dLogDensity.numDesignVariables());
      auto dv = gaussian1dLogDensity.designVariable(0);
      Eigen::MatrixXd p;
      dv->getParameters(p);
      ASSERT_EQ(1, p.size());
      dvValues[i] = p(0,0);
    }

    EXPECT_DOUBLE_EQ((double)sampler.statistics().getNumAcceptedSamples(true)/(double)sampler.statistics().getNumIterations(true),
                     sampler.statistics().getAcceptanceRate(true));

    // check sample mean
//    std::cout << "mean est: " << dvValues.mean() << ", mean true: " << meanTrue << std::endl;
//    std::cout << "var est: " << (dvValues.array() - dvValues.mean()).matrix().squaredNorm()/(dvValues.rows() - 1.0) << ", var true: " << sigmaTrue*sigmaTrue << std::endl;

    EXPECT_NEAR(dvValues.mean(), meanTrue, 4.*sigmaTrue) << "This failure does not necessarily have to be an error. It should just appear "
        " with a probability of 0.00633 %";

    // check sample variance
    EXPECT_NEAR((dvValues.array() - dvValues.mean()).matrix().squaredNorm()/(dvValues.rows() - 1.0), sigmaTrue*sigmaTrue, 1e0) << "This failure does "
        "not necessarily have to be an error. It should just appear very rarely";

    // Run until a specified number of samples was accepted
    sampler.run(numeric_limits<size_t>::max(), 1);
    EXPECT_EQ(sampler.statistics().getNumAcceptedSamples(false), 1);
    EXPECT_GT(sampler.statistics().getAcceptanceRate(false), 0.0);
    EXPECT_GT(sampler.statistics().getNumIterations(false), 0);
    EXPECT_GT(sampler.statistics().getAcceptanceRate(true), 0.0);
    EXPECT_GT(sampler.statistics().getNumIterations(true), 0);

    // Check that re-initializing resets values
    sampler.initialize();
    EXPECT_DOUBLE_EQ(sampler.statistics().getAcceptanceRate(), 0.0);
    EXPECT_EQ(sampler.statistics().getNumIterations(), 0);

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
