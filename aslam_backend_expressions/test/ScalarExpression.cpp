#include <sm/eigen/gtest.hpp>
#include <sm/eigen/NumericalDiff.hpp>
#include <aslam/backend/ScalarExpression.hpp>
#include <sm/kinematics/rotations.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>
#include <aslam/backend/RotationQuaternion.hpp>
#include <aslam/backend/Scalar.hpp>
#include <aslam/backend/RotationExpression.hpp>
#include <sm/random.hpp>

using namespace aslam::backend;
using namespace sm::kinematics;
typedef Eigen::Matrix<double,1,1> Vector1d;

struct ScalarExpressionNodeFunctor
{
    typedef Vector1d value_t;
    typedef value_t::Scalar scalar_t;
    typedef Eigen::VectorXd input_t;
    typedef Eigen::MatrixXd jacobian_t;

  
    ScalarExpressionNodeFunctor(ScalarExpression dv) :
        _dv(dv) {}

    input_t update(const input_t & x, int c, scalar_t delta) { input_t xnew = x; xnew[c] += delta; return xnew; }

    ScalarExpression _dv;

    Eigen::VectorXd operator()(const Eigen::VectorXd & dr)
        {
    
            Vector1d p;
            p(0,0) = _dv.toScalar();
            JacobianContainer J(1);
            _dv.evaluateJacobians(J);

            int offset = 0;
            for(size_t i = 0; i < J.numDesignVariables(); i++)
            {
                DesignVariable * d = J.designVariable(i);
                d->update(&dr[offset],d->minimalDimensions());
                offset += d->minimalDimensions();
            }

            p(0,0) = _dv.toScalar();
 
            for(size_t i = 0; i < J.numDesignVariables(); i++)
            {
                DesignVariable * d = J.designVariable(i);
                d->revertUpdate();
            }

            //std::cout << "returning " << p << std::endl;
            return p;
   
        }
};


void testJacobian(ScalarExpression dv)
{
    ScalarExpressionNodeFunctor functor(dv);
  
    sm::eigen::NumericalDiff<ScalarExpressionNodeFunctor> numdiff(functor);
  
    /// Discern the size of the jacobian container
    Vector1d p;
    p(0,0) = (dv.toScalar());
    JacobianContainer Jc(1);
    dv.evaluateJacobians(Jc);
   
    Eigen::VectorXd dp(Jc.cols());
    dp.setZero();
    Eigen::MatrixXd Jest = numdiff.estimateJacobian(dp);
 
    sm::eigen::assertNear(Jc.asSparseMatrix(), Jest, 1e-6, SM_SOURCE_FILE_POS, "Testing the quat Jacobian");
}



// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testScalarProduct)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        Scalar point2(sm::random::rand());
        point2.setActive(true);
        point2.setBlockIndex(2);
        ScalarExpression p2 = point2.toExpression();

        ScalarExpression p_cross = p1 * p2;

        SCOPED_TRACE("");
        testJacobian(p_cross);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}

// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testScalarProduct2)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        double p2 = sm::random::rand();

        ScalarExpression p_cross = p1 * p2;

        SCOPED_TRACE("");
        testJacobian(p_cross);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}






// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testScalarAddition)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        Scalar point2(sm::random::rand());
        point2.setActive(true);
        point2.setBlockIndex(2);
        ScalarExpression p2 = point2.toExpression();

        ScalarExpression p_add = p1 + p2;

        SCOPED_TRACE("");
        testJacobian(p_add);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}

// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testScalarSubtraction)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        Scalar point2(sm::random::rand());
        point2.setActive(true);
        point2.setBlockIndex(2);
        ScalarExpression p2 = point2.toExpression();

        ScalarExpression p_diff = p1 - p2;

        SCOPED_TRACE("");
        testJacobian(p_diff);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}

// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testVectorSubtraction)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        double p2 = sm::random::rand();

        ScalarExpression p_diff = p1 - p2;

        SCOPED_TRACE("");
        testJacobian(p_diff);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}

// Test that the jacobian matches the finite difference jacobian
TEST(ScalarExpressionNodeTestSuites, testVectorAddition)
{
    try
    {
        using namespace sm::kinematics;
        Scalar point1(sm::random::rand());
        point1.setActive(true);
        point1.setBlockIndex(1);
        ScalarExpression p1 = point1.toExpression();

        double p2 = sm::random::rand();

        ScalarExpression p_diff = p1 + p2;

        SCOPED_TRACE("");
        testJacobian(p_diff);

    }
    catch(std::exception const & e)
    {
        FAIL() << e.what();
    }
}

TEST(ScalarExpressionNodeTestSuites, testVectorOpsFailure)
{
    using namespace aslam::backend;
    // In [12]: p
    // Out[12]: 628.233093262
    double p = 628.233093262;

    // In [13]: t
    // Out[13]: 1338888490.598351
    double t = 1338888490.598351;
    // In [14]: cC.eC.lineDelayDv.toScalar() * p + t
    // Out[14]: 1338888490.6249666
    double ld = 4.23659880956014e-05;

    Scalar ldDv(ld);
    // In [15]: lineDelayDvExpression = cC.eC.lineDelayDv.toExpression()
    ScalarExpression ldExp = ldDv.toExpression();
    ASSERT_EQ(ldDv.toScalar(), ld);

    // In [16]: lineDelayDvExpression.toScalar()
    // Out[16]: 4.23659880956014e-05  
    ASSERT_EQ(ld, ldExp.toScalar());

    // In [17]: keypointOffset = lineDelayDvExpression * p
    ScalarExpression keypointOffset = ldExp * p;

    // In [19]: keypointOffset.toScalar()
    // Out[19]: 0.02661571575038882
    ASSERT_EQ(ld * p, keypointOffset.toScalar());

    // In [20]: keypointTime = keypointOffset + t
    ScalarExpression keypointTime = keypointOffset + t;

    // In [21]: keypointTime.toScalar()
    // Out[21]: 1338888448.0266156
    ASSERT_EQ(ld * p + t, keypointTime.toScalar());

    // In [22]: cC.eC.lineDelayDv.toScalar() * p + t
    // Out[22]: 1338888490.6249666
    ASSERT_EQ(ldDv.toScalar() * p + t, keypointTime.toScalar());


}
