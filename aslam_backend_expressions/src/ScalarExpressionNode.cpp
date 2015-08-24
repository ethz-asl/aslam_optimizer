#include <aslam/backend/ScalarExpressionNode.hpp>
#include <sm/kinematics/rotations.hpp>

namespace aslam {
    namespace backend {
    
        ScalarExpressionNode::ScalarExpressionNode()
        {

        }

        ScalarExpressionNode::~ScalarExpressionNode()
        {
        
        }


        /// \brief Evaluate the scalar matrix.
        double ScalarExpressionNode::toScalar() const
        {
            return toScalarImplementation();
        }

      
        /// \brief Evaluate the Jacobians
        void ScalarExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const
        {
            evaluateJacobiansImplementation(outJacobians);
        }
   
    
        /// \brief Evaluate the Jacobians and apply the chain rule.
        void ScalarExpressionNode::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            SM_ASSERT_EQ_DBG(Exception, applyChainRule.cols(), 1, "The chain rule matrix must have one columns");
            evaluateJacobiansImplementation(outJacobians, applyChainRule);
        }

        void ScalarExpressionNode::getDesignVariables(DesignVariable::set_t & designVariables) const
        {
            getDesignVariablesImplementation(designVariables);
        }


          
        ScalarExpressionNodeMultiply::ScalarExpressionNodeMultiply(boost::shared_ptr<ScalarExpressionNode> lhs,
                                                                   boost::shared_ptr<ScalarExpressionNode> rhs) :
            _lhs(lhs), _rhs(rhs)
        {

        }

        ScalarExpressionNodeMultiply::~ScalarExpressionNodeMultiply()
        {

        }
        double ScalarExpressionNodeMultiply::toScalarImplementation() const
        {
            return _lhs->toScalar() * _rhs->toScalar();
        }

        void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::MatrixXd L(1,1), R(1,1);
            L(0,0) = _lhs->toScalar();
            R(0,0) = _rhs->toScalar();
            _lhs->evaluateJacobians(outJacobians, R);
            _rhs->evaluateJacobians(outJacobians, L);
        }

        void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::MatrixXd L(1,1), R(1,1);
            L(0,0) = _lhs->toScalar();
            R(0,0) = _rhs->toScalar();
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
            _rhs->evaluateJacobians(outJacobians, applyChainRule * L);
        }

        void ScalarExpressionNodeMultiply::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
            _rhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeDivide::ScalarExpressionNodeDivide(boost::shared_ptr<ScalarExpressionNode> lhs,
                                                                   boost::shared_ptr<ScalarExpressionNode> rhs) :
            _lhs(lhs), _rhs(rhs)
        {

        }

        ScalarExpressionNodeDivide::~ScalarExpressionNodeDivide()
        {

        }
        double ScalarExpressionNodeDivide::toScalarImplementation() const
        {
            return _lhs->toScalar() / _rhs->toScalar();
        }

        void ScalarExpressionNodeDivide::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::MatrixXd L(1,1), R(1,1);
            const auto rhs_rec = 1./_rhs->toScalar();
            L(0,0) = rhs_rec;
            R(0,0) = -_lhs->toScalar() * rhs_rec * rhs_rec;
            _lhs->evaluateJacobians(outJacobians, L);
            _rhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeDivide::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::MatrixXd L(1,1), R(1,1);
            const auto rhs_rec = 1./_rhs->toScalar();
            L(0,0) = rhs_rec;
            R(0,0) = -_lhs->toScalar() * rhs_rec * rhs_rec;
            _lhs->evaluateJacobians(outJacobians, applyChainRule * L);
            _rhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeDivide::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
            _rhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodeAdd::ScalarExpressionNodeAdd(boost::shared_ptr<ScalarExpressionNode> lhs,
                                                         boost::shared_ptr<ScalarExpressionNode> rhs,
                                                         double multiplyRhs) :
            _lhs(lhs), _rhs(rhs), _multiplyRhs(multiplyRhs)
        {
            
        }

        ScalarExpressionNodeAdd::~ScalarExpressionNodeAdd()
        {

        }

        double ScalarExpressionNodeAdd::toScalarImplementation() const
        {
            return _lhs->toScalar() + _multiplyRhs * _rhs->toScalar();
        }

        void ScalarExpressionNodeAdd::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::MatrixXd R(1,1);
            R(0,0) = _multiplyRhs;
            _lhs->evaluateJacobians(outJacobians);
            _rhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAdd::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            _lhs->evaluateJacobians(outJacobians, applyChainRule);
            _rhs->evaluateJacobians(outJacobians, applyChainRule * _multiplyRhs);
        }

        void ScalarExpressionNodeAdd::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
            _rhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodeSqrt::ScalarExpressionNodeSqrt(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeSqrt::~ScalarExpressionNodeSqrt()
        {

        }

        double ScalarExpressionNodeSqrt::toScalarImplementation() const
        {
            SM_ASSERT_GT(std::runtime_error, _lhs->toScalar(), 0.0, "");
            return sqrt(_lhs->toScalar());
        }

        void ScalarExpressionNodeSqrt::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = 1./(2.*sqrt(_lhs->toScalar()));
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeSqrt::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = 1./(2.*sqrt(_lhs->toScalar()));
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeSqrt::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeLog::ScalarExpressionNodeLog(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeLog::~ScalarExpressionNodeLog()
        {

        }

        double ScalarExpressionNodeLog::toScalarImplementation() const
        {
            SM_ASSERT_GT(std::runtime_error, _lhs->toScalar(), 0.0, "");
            return log(_lhs->toScalar());
        }

        void ScalarExpressionNodeLog::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = 1./(_lhs->toScalar());
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeLog::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = 1./(_lhs->toScalar());
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeLog::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodeExp::ScalarExpressionNodeExp(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeExp::~ScalarExpressionNodeExp()
        {

        }

        double ScalarExpressionNodeExp::toScalarImplementation() const
        {
            return exp(_lhs->toScalar());
        }

        void ScalarExpressionNodeExp::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = exp(_lhs->toScalar());
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeExp::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = exp(_lhs->toScalar());
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeExp::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodeAtan::ScalarExpressionNodeAtan(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeAtan::~ScalarExpressionNodeAtan()
        {

        }

        double ScalarExpressionNodeAtan::toScalarImplementation() const
        {
            return atan(_lhs->toScalar());
        }

        void ScalarExpressionNodeAtan::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            R(0,0) = 1./(1. + lhss * lhss);
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAtan::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            R(0,0) = 1./(1. + lhss*lhss);
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeAtan::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeAtan2::ScalarExpressionNodeAtan2(boost::shared_ptr<ScalarExpressionNode> lhs, boost::shared_ptr<ScalarExpressionNode> rhs) :
            _lhs(lhs),
            _rhs(rhs)
        {

        }

        ScalarExpressionNodeAtan2::~ScalarExpressionNodeAtan2()
        {

        }

        double ScalarExpressionNodeAtan2::toScalarImplementation() const
        {
            return atan2(_lhs->toScalar(), _rhs->toScalar());
        }

        void ScalarExpressionNodeAtan2::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            // _rhs corresponds to x, _lhs to y
            Eigen::Matrix<double, 1, 1> R(1,1);
            Eigen::Matrix<double, 1, 1> L(1,1);
            const double factor = 1./(_lhs->toScalar()*_lhs->toScalar() + _rhs->toScalar()*_rhs->toScalar());
            R(0,0) = -_lhs->toScalar()*factor;
            L(0,0) = _rhs->toScalar()*factor;
            _lhs->evaluateJacobians(outJacobians, L);
            _rhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAtan2::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            // _rhs corresponds to x, _lhs to y
            Eigen::Matrix<double, 1, 1> R(1,1);
            Eigen::Matrix<double, 1, 1> L(1,1);
            const double factor = 1./(_lhs->toScalar()*_lhs->toScalar() + _rhs->toScalar()*_rhs->toScalar());
            R(0,0) = -_lhs->toScalar()*factor;
            L(0,0) = _rhs->toScalar()*factor;
            _lhs->evaluateJacobians(outJacobians, applyChainRule*L);
            _rhs->evaluateJacobians(outJacobians, applyChainRule*R);
        }

        void ScalarExpressionNodeAtan2::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
            _rhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeAcos::ScalarExpressionNodeAcos(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeAcos::~ScalarExpressionNodeAcos()
        {

        }

        double ScalarExpressionNodeAcos::toScalarImplementation() const
        {
            auto arg = _lhs->toScalar();
            SM_ASSERT_LE(Exception, arg, 1.0, "");
            SM_ASSERT_GE(Exception, arg, -1.0, "");
            return acos(arg);
        }

        void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {

            auto arg = _lhs->toScalar();
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            SM_ASSERT_LE(std::runtime_error, lhss, 1.0, "");
            R(0,0) = -1./sqrt(1. - lhss*lhss);
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            auto arg = _lhs->toScalar();
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            SM_ASSERT_LE(std::runtime_error, lhss, 1.0, "");
            R(0,0) = -1./sqrt(1. - lhss*lhss);
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodeAcos::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _lhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodeAcosSquared::ScalarExpressionNodeAcosSquared(boost::shared_ptr<ScalarExpressionNode> lhs) :
            _lhs(lhs)
        {

        }

        ScalarExpressionNodeAcosSquared::~ScalarExpressionNodeAcosSquared()
        {

        }

        double ScalarExpressionNodeAcosSquared::toScalarImplementation() const
        {
            auto arg = _lhs->toScalar();
            SM_ASSERT_LE(Exception, arg, 1.0, "");
            SM_ASSERT_GE(Exception, arg, -1.0, "");
            auto tmp = acos(arg);
            return tmp*tmp;
        }

        void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            auto arg = _lhs->toScalar();
            Eigen::Matrix<double, 1, 1> R(1,1);
            if (pow((1-arg),4) < std::numeric_limits<double>::min())   // series expansion at x = 1
            {
                auto pow1 = arg - 1.0;
                auto pow2 = pow1 * pow1;
                R(0,0) = -2.0 + 2.0/3.0*pow1 - 4.0/15.0*pow2 + 4.0/35.0*pow1*pow2;
            }
            else
            {
                R(0,0) = -2.0*acos(arg)/sqrt(1.0 - arg*arg);
            }
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            auto arg = _lhs->toScalar();
            Eigen::Matrix<double, 1, 1> R(1,1);
            R(0,0) = -2*acos(arg)/sqrt(1. - arg*arg);
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);

        }

        void ScalarExpressionNodeAcosSquared::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
          _lhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeConstant::ScalarExpressionNodeConstant(double s) : _s(s)
        {
        }

        ScalarExpressionNodeConstant::~ScalarExpressionNodeConstant()
        {
        }

        ScalarExpressionNodeNegated::ScalarExpressionNodeNegated(boost::shared_ptr<ScalarExpressionNode> rhs) :
            _rhs(rhs)
        {
        }

        ScalarExpressionNodeNegated::~ScalarExpressionNodeNegated()
        {
        }

        double ScalarExpressionNodeNegated::toScalarImplementation() const
        {
            return -_rhs->toScalar();
        }

        void ScalarExpressionNodeNegated::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::MatrixXd R(1,1);
            R(0,0) = -1;
            _rhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeNegated::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            _rhs->evaluateJacobians(outJacobians, -applyChainRule);
        }

        void ScalarExpressionNodeNegated::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            _rhs->getDesignVariables(designVariables);
        }
    } // namespace backend
} // namespace aslam
