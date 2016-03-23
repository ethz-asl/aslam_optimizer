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
            return evaluateImplementation();
        }

      
        /// \brief Evaluate the Jacobians
        void ScalarExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const // TODO: inline this method!!!!!!!!! (and everywhere else)
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
        double ScalarExpressionNodeMultiply::evaluateImplementation() const
        {
            return _lhs->toScalar() * _rhs->toScalar();
        }

        void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> L(1,1), R(1,1);
            L(0,0) = _lhs->toScalar();
            R(0,0) = _rhs->toScalar();

            _lhs->evaluateJacobians(outJacobians.apply(R));
            _rhs->evaluateJacobians(outJacobians.apply(L));
        }

        void ScalarExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> L(1,1), R(1,1);
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
        double ScalarExpressionNodeDivide::evaluateImplementation() const
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

        double ScalarExpressionNodeAdd::evaluateImplementation() const
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

        double ScalarExpressionNodeSqrt::evaluateImplementation() const
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

        double ScalarExpressionNodeLog::evaluateImplementation() const
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

        double ScalarExpressionNodeExp::evaluateImplementation() const
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

        double ScalarExpressionNodeAtan::evaluateImplementation() const
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

        double ScalarExpressionNodeAtan2::evaluateImplementation() const
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

        double ScalarExpressionNodeAcos::evaluateImplementation() const
        {
            auto lhss = _lhs->toScalar();
            SM_ASSERT_LE(Exception, lhss, 1.0, "");
            SM_ASSERT_GE(Exception, lhss, -1.0, "");
            return acos(lhss);
        }

        void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            SM_ASSERT_LE(std::runtime_error, lhss, 1.0, "");
            R(0,0) = -1./sqrt(1. - lhss*lhss);
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAcos::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
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

        double ScalarExpressionNodeAcosSquared::evaluateImplementation() const
        {
            const auto lhss = _lhs->toScalar();
            SM_ASSERT_LE(Exception, lhss, 1.0, "");
            SM_ASSERT_GE(Exception, lhss, -1.0, "");
            auto tmp = acos(lhss);
            return tmp*tmp;
        }

        void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            auto pow1 = lhss - 1.0;
            auto pow2 = pow1 * pow1;

            if (pow2 < std::numeric_limits<double>::epsilon())   // series expansion at x = 1
            {
                R(0,0) = -2.0 + 2.0/3.0*pow1;
            }
            else
            {
                R(0,0) = -2.0*acos(lhss)/sqrt(1.0 - lhss*lhss);
            }
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeAcosSquared::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            auto pow1 = lhss - 1.0;
            auto pow2 = pow1 * pow1;

            if (pow2 < std::numeric_limits<double>::epsilon())   // series expansion at x = 1
            {
                R(0,0) = -2.0 + 2.0/3.0*pow1;
            }
            else
            {
                R(0,0) = -2.0*acos(lhss)/sqrt(1.0 - lhss*lhss);
            }
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);

        }

        void ScalarExpressionNodeAcosSquared::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
          _lhs->getDesignVariables(designVariables);
        }

        ScalarExpressionNodeInverseSigmoid::ScalarExpressionNodeInverseSigmoid(boost::shared_ptr<ScalarExpressionNode> lhs,
                                                                 const double height,
                                                                 const double scale,
                                                                 const double shift)
          : _lhs(lhs), _height(height), _scale(scale), _shift(shift)
        {

        }

        ScalarExpressionNodeInverseSigmoid::~ScalarExpressionNodeInverseSigmoid()
        {

        }

        double ScalarExpressionNodeInverseSigmoid::evaluateImplementation() const
        {
            const auto lhss = _lhs->toScalar();
            return _height / (exp((lhss - _shift) * _scale) + 1.0);
        }

        void ScalarExpressionNodeInverseSigmoid::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            const double threshold = 50;

            if (lhss > threshold) {
              // approximate with Taylor expansion since exponents become too big otherwise
              auto den = exp(_scale*_shift) + exp(_scale*threshold);
              auto denSq = den*den;
              R(0,0) = - _height*_scale*exp(_scale*(threshold+_shift)) / denSq +
                       _height*_scale*_scale*(lhss-threshold)*exp(_scale*(_shift+threshold))
                       *(exp(threshold*_scale) - exp(_scale*_shift)) / (denSq*den);
            }
            else {
              auto den = 1 + exp(_scale*(lhss-_shift));
              R(0,0) = - _height*_scale*exp(_scale*(lhss-_shift)) / (den * den);
            }

            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodeInverseSigmoid::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
          Eigen::Matrix<double, 1, 1> R(1,1);
          const auto lhss = _lhs->toScalar();
          const double threshold = 50;

          if (lhss > threshold) {
            // approximate with Taylor expansion since exponents become too big otherwise
            auto den = exp(_scale*_shift) + exp(_scale*threshold);
            auto denSq = den*den;
            R(0,0) = - _height*_scale*exp(_scale*(threshold+_shift)) / denSq +
                     _height*_scale*_scale*(lhss-threshold)*exp(_scale*(_shift+threshold))
                     *(exp(threshold*_scale) - exp(_scale*_shift)) / (denSq*den);
          }
          else {
            auto den = 1 + exp(_scale*(lhss-_shift));
            R(0,0) = - _height*_scale*exp(_scale*(lhss-_shift)) / (den * den);
          }

            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);

        }

        void ScalarExpressionNodeInverseSigmoid::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
          _lhs->getDesignVariables(designVariables);
        }


        ScalarExpressionNodePower::ScalarExpressionNodePower(boost::shared_ptr<ScalarExpressionNode> lhs, const int k)
          :_lhs(lhs), _power(k)
        {

        }

        ScalarExpressionNodePower::~ScalarExpressionNodePower()
        {

        }

        double ScalarExpressionNodePower::evaluateImplementation() const
        {
            const auto lhss = _lhs->toScalar();
            return pow(lhss, _power);
        }

        void ScalarExpressionNodePower::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            R(0,0) = _power * pow(lhss, _power-1);
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, R);
        }

        void ScalarExpressionNodePower::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            Eigen::Matrix<double, 1, 1> R(1,1);
            const auto lhss = _lhs->toScalar();
            R(0,0) = _power * pow(lhss, _power-1);
            SM_ASSERT_FALSE(Exception, std::isnan(R(0,0)), "");
            _lhs->evaluateJacobians(outJacobians, applyChainRule * R);
        }

        void ScalarExpressionNodePower::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
          _lhs->getDesignVariables(designVariables);
        }


        ScalarExpressionPiecewiseExpression::ScalarExpressionPiecewiseExpression(boost::shared_ptr<ScalarExpressionNode> e1, boost::shared_ptr<ScalarExpressionNode> e2, std::function<bool()> useFirst)
          : _e1(e1), _e2(e2), _useFirst(useFirst)
        {

        }

        ScalarExpressionPiecewiseExpression::~ScalarExpressionPiecewiseExpression()
        {

        }

        double ScalarExpressionPiecewiseExpression::evaluateImplementation() const
        {
          if (_useFirst()) {
            return _e1->toScalar();
          } else {
            return _e2->toScalar();
          }
        }

        void ScalarExpressionPiecewiseExpression::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
          if (_useFirst()) {
            return _e1->evaluateJacobians(outJacobians);
          } else {
            return _e2->evaluateJacobians(outJacobians);
          }
        }

        void ScalarExpressionPiecewiseExpression::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
          if (_useFirst()) {
            return _e1->evaluateJacobians(outJacobians, applyChainRule);
          } else {
            return _e2->evaluateJacobians(outJacobians, applyChainRule);
          }
        }

        void ScalarExpressionPiecewiseExpression::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
          if (_useFirst()) {
            return _e1->getDesignVariables(designVariables);
          } else {
            return _e2->getDesignVariables(designVariables);
          }
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

        double ScalarExpressionNodeNegated::evaluateImplementation() const
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
