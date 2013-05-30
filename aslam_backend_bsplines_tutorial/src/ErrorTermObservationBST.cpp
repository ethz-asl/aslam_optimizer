#include <aslam/backend/ErrorTermObservationBST.hpp>

namespace aslam {
  namespace backend {
    
    
    ErrorTermObservationBST::ErrorTermObservationBST(aslam::backend::VectorExpression<1> robotPos, aslam::backend::EuclideanExpression wallPoint, double y, double sigma2_n) :
		_robotPos(robotPos), _wallPoint(wallPoint), _y(y)
    {
        Eigen::Matrix<double,1,1> invR;
        invR(0,0) = (1.0/sigma2_n);
      // Fill in the inverse covariance. In this scalar case, this is just an inverse variance.
        setInvR( invR );

      // Tell the super class about the design variables:
        JacobianContainer::set_t dvs;
        robotPos.getDesignVariables(dvs);
        std::vector<aslam::backend::DesignVariable*> designVariablePtrs;
        for(JacobianContainer::set_t::iterator it = dvs.begin(); it != dvs.end(); ++it)
        {
        	designVariablePtrs.push_back(*it);
        }
        wallPoint.getDesignVariables(dvs);
        for(JacobianContainer::set_t::iterator it = dvs.begin(); it != dvs.end(); ++it)
        {
        	designVariablePtrs.push_back(*it);
        }
        ErrorTermFs<1>::setDesignVariables(designVariablePtrs);
    }

    ErrorTermObservationBST::~ErrorTermObservationBST()
    {

    }


    /// \brief evaluate the error term and return the weighted squared error e^T invR e
    double ErrorTermObservationBST::evaluateErrorImplementation()
    {
      // Build the error from the measurement _y and the design variables

    	double wallPosition = _wallPoint.toEuclidean()(0);
    	double robotPosition = _robotPos.evaluate()(0);
        error_t error;
        error(0) = _y - 1.0/(wallPosition - robotPosition);
        setError(error);
        return evaluateChiSquaredError();
    }


    /// \brief evaluate the jacobians
    void ErrorTermObservationBST::evaluateJacobiansImplementation()
    {
      double wallPosition = _wallPoint.toEuclidean()(0);
      double robotPosition = _robotPos.evaluate()(0);
      double hat_y = -1.0/(wallPosition - robotPosition);
      Eigen::MatrixXd hat_y2(1,1);
      hat_y2(0,0) = hat_y * hat_y;

      _robotPos.evaluateJacobians(ErrorTermFs<1>::_jacobians, -hat_y2);
      _wallPoint.evaluateJacobians(ErrorTermFs<1>::_jacobians, hat_y2);

//      _jacobians.add(_x_k, -hat_y2);
//      _jacobians.add(_w, hat_y2);
    }


  } // namespace backend
} // namespace aslam
