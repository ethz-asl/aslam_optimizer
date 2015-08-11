#include <sm/eigen/gtest.hpp>
#include <sm/timing/Timer.hpp>
#include <sm/random.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/SamplerMetropolisHastings.hpp>
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


TEST(OptimizerSamplerMcmcTestSuite, testSamplerMcmc)
{
  try {
    typedef OptimizationProblem LogDensity;
    typedef boost::shared_ptr<LogDensity> LogDensityPtr;

    sm::random::seed(std::time(nullptr));

    LogDensityPtr gaussian1dLogDensityPtr(new LogDensity);
    OptimizationProblem& gaussian1dLogDensity = *gaussian1dLogDensityPtr;

    const double meanTrue = 10.0;
    const double sigmaTrue = 2.0;
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

    // Initialize and test options
    sm::BoostPropertyTree pt;
    pt.setDouble("transitionKernelSigma", 1.0);
    SamplerMetropolisHastingsOptions options(pt);
    EXPECT_DOUBLE_EQ(options.transitionKernelSigma, pt.getDouble("transitionKernelSigma"));

    // Set and test log density
    SamplerMetropolisHastings sampler(options);
    sampler.setNegativeLogDensity(gaussian1dLogDensityPtr);
    EXPECT_NO_THROW(sampler.checkNegativeLogDensitySetup());
    EXPECT_DOUBLE_EQ(0.0, sampler.statistics().getAcceptanceRate());
    EXPECT_EQ(0, sampler.statistics().getNumIterations());

    // Parameters
    const int nSamples = 1000;
    const int nStepsBurnIn = 100;
    const int nStepsSkip = 50;

    // Burn-in
    sampler.run(nStepsBurnIn);
    EXPECT_GE(sampler.statistics().getAcceptanceRate(), 1e-3);
    EXPECT_LE(sampler.statistics().getAcceptanceRate(), 1.0);
    EXPECT_EQ(nStepsBurnIn, sampler.statistics().getNumIterations());

    // Now let's retrieve samples
    Eigen::VectorXd dvValues(nSamples);
    for (size_t i=0; i<nSamples; i++) {
      sampler.run(nStepsSkip);
      EXPECT_GE(sampler.statistics().getAcceptanceRate(), 0.0);
      EXPECT_LE(sampler.statistics().getAcceptanceRate(), 1.0);
      EXPECT_EQ(sampler.statistics().getNumIterations(), nStepsSkip);
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
    EXPECT_GT(sampler.statistics().getAcceptanceRate(), 0.0);
    EXPECT_GT(sampler.statistics().getNumIterations(), 0);

    // Check that re-initializing resets values
    sampler.initialize();
    EXPECT_DOUBLE_EQ(0.0, sampler.statistics().getAcceptanceRate());
    EXPECT_EQ(0, sampler.statistics().getNumIterations());

#ifdef aslam_backend_ENABLE_TIMING
    sm::timing::Timing::print(cout, sm::timing::SORT_BY_TOTAL);
#endif

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
