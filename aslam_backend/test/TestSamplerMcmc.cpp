#include <sm/eigen/gtest.hpp>
#include <sm/timing/Timer.hpp>
#include <sm/random.hpp>
#include <aslam/backend/SamplerMcmc.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/test/ErrorTermTester.hpp>
#include "SampleDvAndError.hpp"

using namespace std;
using namespace boost;
using namespace aslam::backend;

/// \brief Encodes the error \f$ -\frac{\left(\mathbf x - \mathbf \mu\right)^2}{2.0 \sigma^2}\f$
class GaussianLogDensityError : public ScalarNonSquaredErrorTerm {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef ScalarNonSquaredErrorTerm parent_t;

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
    pt.setDouble("transitionKernelSigma", 1.0);
    SamplerMcmcOptions options(pt);
    EXPECT_DOUBLE_EQ(pt.getDouble("transitionKernelSigma"), options.transitionKernelSigma);

    // Set and test log density
    SamplerMcmc sampler(options);
    sampler.setLogDensity(gaussian1dLogDensityPtr);
    EXPECT_NO_THROW(sampler.checkLogDensitySetup());

    // Parameters
    const int nSamples = 1000;
    const int nStepsBurnIn = 100;
    const int nStepsSkip = 50;

    // Burn-in
    sampler.run(nStepsBurnIn);

    // Now let's retrieve samples
    Eigen::VectorXd dvValues(nSamples);
    for (size_t i=0; i<nSamples; i++) {
      sampler.run(nStepsSkip);
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
    EXPECT_NEAR((dvValues.array() - dvValues.mean()).matrix().squaredNorm()/(dvValues.rows() - 1.0), sigmaTrue*sigmaTrue, 3e-1) << "This failure does "
        "not necessarily have to be an error. It should just appear very rarely";

    std::ostringstream os;
    sm::timing::Timing::print(os);
    std::cout << os.str() << std::endl;

  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
