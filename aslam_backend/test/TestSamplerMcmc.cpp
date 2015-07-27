#include <sm/eigen/gtest.hpp>
#include <sm/timing/Timer.hpp>
#include <sm/random.hpp>
#include <aslam/backend/SamplerMcmc.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"


/// \brief Encodes the error \f$ -\frac{\left(\mathbf x - \mathbf \mu\right)^2}{2.0 \sigma^2}\f$
class GaussianLogDensityError : public aslam::backend::ScalarNonSquaredErrorTerm {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef aslam::backend::ScalarNonSquaredErrorTerm parent_t;

  Scalar* _x; /// \brief The design variable
  Scalar::Vector1d _mu;  /// \brief The mean
  Scalar::Vector1d _varInv;  /// \brief The inverse variance

  GaussianLogDensityError(Scalar* x) : _x(x) {
    _x->setActive(true);
    parent_t::setDesignVariables(_x);
    setWeight(1.0);
    _mu << 0.0;
    _varInv << 1.0;
  }
  virtual ~GaussianLogDensityError() {}

  void setMean(const double mu) { _mu << mu; }
  void setVariance(const double var) { _varInv << 1./var; }

  /// \brief evaluate the error term
  virtual double evaluateErrorImplementation() {
    const Scalar::Vector1d dv = (_x->_v - _mu);
    return -0.5*(_varInv*dv*dv)[0];
  }

  /// \brief evaluate the jacobian
  virtual void evaluateJacobiansImplementation(aslam::backend::JacobianContainer & outJ) {
    outJ.add( _x, -(_x->_v - _mu)*_varInv );
  }

};

TEST(OptimizerSamplerMcmcTestSuite, testSamplerMcmc)
{
  try {
    using namespace aslam::backend;
    typedef OptimizationProblem LogDensity;
    typedef boost::shared_ptr<LogDensity> LogDensityPtr;

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
    boost::shared_ptr<GaussianLogDensityError> err(new GaussianLogDensityError(sdv.get()));
    err->setMean(meanTrue);
    err->setVariance(sigmaTrue*sigmaTrue);
    gaussian1dLogDensity.addErrorTerm(err);
    SCOPED_TRACE("");
    testErrorTerm(err);

    // Initialize and test options
    sm::BoostPropertyTree pt;
    pt.setInt("nSamples", 1000);
    pt.setDouble("nStepsBurnIn", 100);
    pt.setDouble("nStepsSkip", 50);
    pt.setDouble("transitionKernelSigma", 1.0);
    SamplerMcmcOptions options(pt);
    EXPECT_EQ(pt.getInt("nSamples"), options.nSamples);
    EXPECT_EQ(pt.getInt("nStepsBurnIn"), options.nStepsBurnIn);
    EXPECT_EQ(pt.getInt("nStepsSkip"), options.nStepsSkip);
    EXPECT_DOUBLE_EQ(pt.getDouble("transitionKernelSigma"), options.transitionKernelSigma);

    // Set and test log density
    SamplerMcmc sampler(options);
    sampler.setLogDensity(gaussian1dLogDensityPtr);
    EXPECT_NO_THROW(sampler.checkLogDensitySetup());

    // Now let's sample.
    Eigen::MatrixXd samples = sampler.run();
    EXPECT_EQ(options.nSamples, samples.cols());
    EXPECT_EQ(1, samples.rows());
    EXPECT_TRUE(samples.allFinite());

    // check sample mean
    EXPECT_NEAR(samples.mean(), meanTrue, 4.*sigmaTrue) << "This failure does not necessarily have to be an error. It should just appear "
        " with a probability of 0.00633 %";

    // check sample variance
    EXPECT_NEAR((samples.array() - samples.mean()).matrix().squaredNorm()/(samples.cols() - 1.0), sigmaTrue*sigmaTrue, 3e-1) << "This failure does "
        "not necessarily have to be an error. It should just appear very rarely";

//    std::ostringstream os;
//    sm::timing::Timing::print(os);
//    std::cout << os.str() << std::endl;

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
