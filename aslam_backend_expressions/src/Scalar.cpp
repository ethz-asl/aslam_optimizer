#include <aslam/backend/Scalar.hpp>


namespace aslam {
    namespace backend {
        Scalar::Scalar(const double & p) :
            _p(p), _p_p(p)
        {

        }
        Scalar::~Scalar()
        {

        }

        /// \brief Revert the last state update.
        void Scalar::revertUpdateImplementation()
        {
            _p = _p_p;
        }
    
        /// \brief Update the design variable.
        void Scalar::updateImplementation(const double * dp, int size)
        {
            _p_p = _p;
      
            _p += dp[0];

        }
    
        /// \brief the size of an update step
        int Scalar::minimalDimensionsImplementation() const
        {
            return 1;
        }
    
        double Scalar::toScalarImplementation() const
        {
            return _p;
        }
    
        void Scalar::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
        {
            Eigen::Matrix<double,1,1> J;
            J(0,0) = 1;
            outJacobians.add(const_cast<Scalar *>(this), J);
        }
    
        void Scalar::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
        {
            outJacobians.add(const_cast<Scalar *>(this), applyChainRule);
        }

        void Scalar::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
        {
            designVariables.insert(const_cast<Scalar *>(this));
        }

        ScalarExpression Scalar::toExpression()
        {
            return ScalarExpression(this);
        }

        Eigen::MatrixXd Scalar::getParameters()
        {
        	Eigen::Matrix<double, 1,1> M;
        	M << toScalar();
        	return M;
        }
      

    } // namespace backend
} // namespace aslam
