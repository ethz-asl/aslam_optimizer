#include <sm/eigen/gtest.hpp>
#include <sm/random.hpp>
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



TEST(VectorExpressionNodeTestSuites, testVectorStackingFromScalars) {
  try {
    Scalar scalar1(sm::random::rand());
    Scalar scalar2(sm::random::rand());
    Scalar scalar3(sm::random::rand());

    VectorExpression<1> singleStacked(scalar1.toExpression());
    VectorExpression<2> twoStacked(scalar1.toExpression(), scalar2.toExpression());
    VectorExpression<3> threeStacked(scalar1.toExpression(), scalar2.toExpression(), scalar3.toExpression());

    testExpression(singleStacked, 1);
    testExpression(twoStacked, 2);
    testExpression(threeStacked, 3);
  }
  catch(std::exception const & e)
  {
    FAIL() << e.what();
  }
}


TEST(VectorExpressionNodeTestSuites, testVectorAdd) {
  try {
    typedef VectorExpression<4> VEC;
    DesignVariableVector<4> vec1(VEC::value_t::Random());
    DesignVariableVector<4> vec2(VEC::value_t::Random());
    const auto vecExp1 = vec1.toExpression();
    const auto vecExp2 = vec2.toExpression();
    const auto sum = vecExp1 + vecExp2;

    testExpression(vecExp1, 1);
    testExpression(vecExp2, 1);
    testExpression(sum, 2);
  }
  catch(std::exception const & e)
  {
    FAIL() << e.what();
  }
}

TEST(VectorExpressionNodeTestSuites, testVectorMultiply) {
  try {
    typedef VectorExpression<4> VEC;
    DesignVariableVector<4> vec(VEC::value_t::Random());
    const auto vecExp = vec.toExpression();
    Scalar point(sm::random::rand());
    const auto pExp = point.toExpression();
    const auto prod = p1 * vecExp;

    testExpression(vecExp, 1);
    testExpression(pExp, 1);
    testExpression(prod, 2);
  } catch (std::exception const& e) {
    FAIL() << e.what();
  }
}
