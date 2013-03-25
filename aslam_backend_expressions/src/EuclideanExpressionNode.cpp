#include <aslam/backend/EuclideanExpressionNode.hpp>
#include <sm/kinematics/rotations.hpp>

namespace aslam {
  namespace backend {
    
    EuclideanExpressionNode::EuclideanExpressionNode()
    {

    }

    EuclideanExpressionNode::~EuclideanExpressionNode()
    {

    }


    /// \brief Evaluate the euclidean matrix.
    Eigen::Vector3d EuclideanExpressionNode::toEuclidean()
    {
      return toEuclideanImplementation();
    }

      
    /// \brief Evaluate the Jacobians
    void EuclideanExpressionNode::evaluateJacobians(JacobianContainer & outJacobians) const
    {
      evaluateJacobiansImplementation(outJacobians);
    }
   
    
    /// \brief Evaluate the Jacobians and apply the chain rule.
    void EuclideanExpressionNode::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      SM_ASSERT_EQ_DBG(Exception, applyChainRule.cols(), 3, "The chain rule matrix must have three columns");
      evaluateJacobiansImplementation(outJacobians, applyChainRule);
    }

    void EuclideanExpressionNode::getDesignVariables(DesignVariable::set_t & designVariables) const
    {
      getDesignVariablesImplementation(designVariables);
    }


    EuclideanExpressionNodeMultiply::EuclideanExpressionNodeMultiply(boost::shared_ptr<RotationExpressionNode> lhs, boost::shared_ptr<EuclideanExpressionNode> rhs) :
         _lhs(lhs), _rhs(rhs)
       {

    	_C_lhs = _lhs->toRotationMatrix();
         _p_rhs = _rhs->toEuclidean();
       }

    EuclideanExpressionNodeMultiply::~EuclideanExpressionNodeMultiply()
    {

    }


    void EuclideanExpressionNodeMultiply::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }


    Eigen::Vector3d EuclideanExpressionNodeMultiply::toEuclideanImplementation()
    {
      _C_lhs = _lhs->toRotationMatrix();
      _p_rhs = _rhs->toEuclidean();

      return _C_lhs * _p_rhs;
    }

    void EuclideanExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, sm::kinematics::crossMx(_C_lhs * _p_rhs));
      _rhs->evaluateJacobians(outJacobians, _C_lhs);
    }

    void EuclideanExpressionNodeMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, applyChainRule * sm::kinematics::crossMx(_C_lhs * _p_rhs));
      _rhs->evaluateJacobians(outJacobians, applyChainRule * _C_lhs);
    }
    // -------------------------------------------------------
    // ## New Class for rotations with MatrixExpressions
    EuclideanExpressionNodeMatrixMultiply::EuclideanExpressionNodeMatrixMultiply(boost::shared_ptr<MatrixExpressionNode> lhs, boost::shared_ptr<EuclideanExpressionNode> rhs) :
         _lhs(lhs), _rhs(rhs)
       {

    	 _A_lhs = _lhs->toFullMatrix();
         _p_rhs = _rhs->toEuclidean();
       }

    EuclideanExpressionNodeMatrixMultiply::~EuclideanExpressionNodeMatrixMultiply()
    {
      
    }


    void EuclideanExpressionNodeMatrixMultiply::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }

    
    Eigen::Vector3d EuclideanExpressionNodeMatrixMultiply::toEuclideanImplementation()
    {
      _A_lhs = _lhs->toFullMatrix();
      _p_rhs = _rhs->toEuclidean();

      return _A_lhs * _p_rhs;
    }

    void EuclideanExpressionNodeMatrixMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
    	double p1 = _p_rhs(0), p2 = _p_rhs(1), p3 = _p_rhs(2);
    	Eigen::Matrix<double, 3,9> J_full;
    	J_full << 
            p1, 0,  0, p2,  0,  0, p3,  0,  0,
            0, p1,  0,  0, p2,  0,  0, p3,  0, 
            0,  0, p1,  0,  0, p2,  0,  0, p3;

        _lhs->evaluateJacobians(outJacobians, J_full); 				 // ## Set in the full 3x9 jacobian matrix
        _rhs->evaluateJacobians(outJacobians, _A_lhs);
        //_rhs->evaluateJacobians(outJacobians, Eigen::Matrix3d::Zero());
    }

    void EuclideanExpressionNodeMatrixMultiply::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
    	double p1 = _p_rhs(0), p2 = _p_rhs(1), p3 = _p_rhs(2);
    	Eigen::Matrix<double, 3,9> J_full;
    	J_full << p1, 0, 0, p2, 0, 0, p3, 0, 0, 0, p1, 0, 0, p2, 0, 0, p3, 0, 0, 0, p1, 0, 0, p2, 0, 0, p3;

        _lhs->evaluateJacobians(outJacobians, applyChainRule * J_full);		        // ## Set in the full  3x9 jacobian matrix
        _rhs->evaluateJacobians(outJacobians, applyChainRule * _A_lhs);
        // _rhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix3d::Identity());
    }

    // ----------------------------

    EuclideanExpressionNodeCrossEuclidean::EuclideanExpressionNodeCrossEuclidean(boost::shared_ptr<EuclideanExpressionNode> lhs, boost::shared_ptr<EuclideanExpressionNode> rhs) :
      _lhs(lhs), _rhs(rhs)
    {

    }

    EuclideanExpressionNodeCrossEuclidean::~EuclideanExpressionNodeCrossEuclidean()
    {

    }


    void EuclideanExpressionNodeCrossEuclidean::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }


    Eigen::Vector3d EuclideanExpressionNodeCrossEuclidean::toEuclideanImplementation()
    {
      return sm::kinematics::crossMx(_lhs->toEuclidean()) * _rhs->toEuclidean();;
    }

    void EuclideanExpressionNodeCrossEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, - sm::kinematics::crossMx(_rhs->toEuclidean()));
      _rhs->evaluateJacobians(outJacobians, sm::kinematics::crossMx(_lhs->toEuclidean()));
    }

    void EuclideanExpressionNodeCrossEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, - applyChainRule * sm::kinematics::crossMx(_rhs->toEuclidean()));
      _rhs->evaluateJacobians(outJacobians, applyChainRule * sm::kinematics::crossMx(_lhs->toEuclidean()));
    }





    EuclideanExpressionNodeAddEuclidean::EuclideanExpressionNodeAddEuclidean(boost::shared_ptr<EuclideanExpressionNode> lhs, boost::shared_ptr<EuclideanExpressionNode> rhs) :
      _lhs(lhs), _rhs(rhs)
    {

    }

    EuclideanExpressionNodeAddEuclidean::~EuclideanExpressionNodeAddEuclidean()
    {

    }


    void EuclideanExpressionNodeAddEuclidean::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }


    Eigen::Vector3d EuclideanExpressionNodeAddEuclidean::toEuclideanImplementation()
    {
      return _lhs->toEuclidean() + _rhs->toEuclidean();
    }

    void EuclideanExpressionNodeAddEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, Eigen::Matrix3d::Identity());
      _rhs->evaluateJacobians(outJacobians, Eigen::Matrix3d::Identity());
    }

    void EuclideanExpressionNodeAddEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix3d::Identity());
      _rhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix3d::Identity());
    }





    EuclideanExpressionNodeSubtractEuclidean::EuclideanExpressionNodeSubtractEuclidean(boost::shared_ptr<EuclideanExpressionNode> lhs, boost::shared_ptr<EuclideanExpressionNode> rhs) :
      _lhs(lhs), _rhs(rhs)
    {

    }

    EuclideanExpressionNodeSubtractEuclidean::~EuclideanExpressionNodeSubtractEuclidean()
    {

    }


    void EuclideanExpressionNodeSubtractEuclidean::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
      _rhs->getDesignVariables(designVariables);
    }


    Eigen::Vector3d EuclideanExpressionNodeSubtractEuclidean::toEuclideanImplementation()
    {
      return _lhs->toEuclidean() - _rhs->toEuclidean();
    }

    void EuclideanExpressionNodeSubtractEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, Eigen::Matrix3d::Identity());
      _rhs->evaluateJacobians(outJacobians, -Eigen::Matrix3d::Identity());
    }

    void EuclideanExpressionNodeSubtractEuclidean::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix3d::Identity());
      _rhs->evaluateJacobians(outJacobians, - applyChainRule * Eigen::Matrix3d::Identity());
    }





    EuclideanExpressionNodeSubtractVector::EuclideanExpressionNodeSubtractVector(boost::shared_ptr<EuclideanExpressionNode> lhs, Eigen::Vector3d rhs) :
      _lhs(lhs), _rhs(rhs)
    {

    }

    EuclideanExpressionNodeSubtractVector::~EuclideanExpressionNodeSubtractVector()
    {

    }


    void EuclideanExpressionNodeSubtractVector::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      _lhs->getDesignVariables(designVariables);
    }


    Eigen::Vector3d EuclideanExpressionNodeSubtractVector::toEuclideanImplementation()
    {
      return _lhs->toEuclidean() - _rhs;
    }

    void EuclideanExpressionNodeSubtractVector::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      _lhs->evaluateJacobians(outJacobians, Eigen::Matrix3d::Identity());
    }

    void EuclideanExpressionNodeSubtractVector::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      _lhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix3d::Identity());
    }



  } // namespace backend
} // namespace aslam