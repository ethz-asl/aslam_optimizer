#include <gtest/gtest.h>

#include <aslam/backend/L1Regularizer.hpp>

using namespace std;
using namespace aslam::backend;

TEST(AslamVChargeBackendTestSuite, testL1Regularizer)
{
  try {

    Scalar dv1(-2.0);
    Scalar dv2(3.0);
    Scalar dv3(0.0);
    vector<Scalar*> dvs;
    dvs.push_back(&dv1);
    dvs.push_back(&dv2);
    dvs.push_back(&dv3);
    for (size_t i=0; i<dvs.size(); i++) {
      dvs[i]->setActive(true);
      dvs[i]->setBlockIndex(i);
    }
    L1Regularizer reg(dvs, 2.0);

    SCOPED_TRACE("");
    double error = reg.evaluateError();
    EXPECT_DOUBLE_EQ(10.0, error);

    reg.setBeta(3.0);
    SCOPED_TRACE("");
    error = reg.evaluateError();
    EXPECT_DOUBLE_EQ(15.0, error);

    Eigen::MatrixXd Jexp(1,3);
    Jexp << -3.0, 3.0, 0.0;
    JacobianContainer jc(1);
    reg.evaluateJacobians(jc);
    Eigen::MatrixXd J = jc.asDenseMatrix();
    EXPECT_TRUE(J.isApprox(Jexp)) << "Computed: " << J << std::endl <<
        "Expected: " << Jexp;
  }
  catch(const std::exception & e)
  {
    FAIL() << e.what();
  }
}
