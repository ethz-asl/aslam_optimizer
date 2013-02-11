#include <aslam/backend/ErrorTermPrior.hpp>

namespace aslam {
  namespace backend {
    ErrorTermPrior::ErrorTermPrior(ScalarDesignVariable * x, double hat_x, double sigma2_x)
      : _x(x), _hat_x(hat_x)
    {
      // Fill in the inverse covariance. In this scalar case, this is just an inverse variance.
      _invR(0,0) = 1.0/sigma2_x;

      // Tell the super class about the design variables:
      setDesignVariables(_x);

    }

    ErrorTermPrior::~ErrorTermPrior()
    {

    }

      
    /// \brief evaluate the error term and return the weighted squared error e^T invR e
    double ErrorTermPrior::evaluateErrorImplementation()
    {
      _error(0) = _x->value() - _hat_x;
      return _error.dot(_invR * _error);	    
    }


    /// \brief evaluate the jacobian
    void ErrorTermPrior::evaluateJacobiansImplementation()
    {
      _jacobians.add(_x, Eigen::MatrixXd::Identity(1,1));
    }


  } // namespace backend
} // namespace aslam
