#include <sm/eigen/gtest.hpp>
#include <Eigen/Geometry>
#include <aslam/backend/Scalar.hpp>
#include <aslam/backend/VectorExpression.hpp>
#include <aslam/backend/DesignVariableVector.hpp>
#include <aslam/backend/test/ExpressionTests.hpp>

using namespace aslam::backend;
using namespace std;

TEST(VectorExpressionNodeTestSuites, testVectorBasicOperations) {
  try {
    const int VEC_ROWS = 5;
    const int VEC_ROWS2 = 3;
    const int VEC_ROWS3 = 2;

    typedef VectorExpression<VEC_ROWS> VEC;
    typedef VectorExpression<Eigen::Dynamic> VEC2;
    typedef VectorExpression<Eigen::Dynamic> VEC3;

    DesignVariableVector<VEC_ROWS> vec(VEC::value_t::Random());
    VEC2::value_t vec2 = VEC2::value_t::Random(VEC_ROWS2);
    DesignVariableVector<Eigen::Dynamic> vec3(VEC3::value_t::Random(VEC_ROWS3));
    VEC vecExp(&vec);
    VEC2 vecExp2(vec2);
    auto vecExp3 = vec3.toExpression();

    EXPECT_EQ(VEC_ROWS, vec.getSize());
    EXPECT_EQ(VEC_ROWS, vecExp.getSize());
    EXPECT_EQ(VEC_ROWS2, vecExp2.getSize());
    EXPECT_EQ(VEC_ROWS3, vec3.getSize());
    EXPECT_EQ(VEC_ROWS3, vecExp3.getSize());

    sm::eigen::assertNear(vecExp.evaluate(), vec.value(), 1e-14, SM_SOURCE_FILE_POS, "Testing evaluation fits initialization.");
    sm::eigen::assertNear(vecExp2.evaluate(), vec2, 1e-14, SM_SOURCE_FILE_POS, "Testing evaluation fits initialization.");
    sm::eigen::assertNear(vecExp3.evaluate(), vec3.value(), 1e-14, SM_SOURCE_FILE_POS, "Testing evaluation fits initialization.");

    testExpression(vecExp, 1);
    testExpression(vecExp2, 0);
    testExpression(vecExp3, 1);
  }
  catch(std::exception const & e)
  {
    FAIL() << e.what();
  }
}
