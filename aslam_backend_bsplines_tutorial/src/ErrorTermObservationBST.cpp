#include <aslam/backend/ErrorTermObservationBST.hpp>
#include <aslam/backend/Scalar.hpp>

namespace aslam {
  namespace backend {
    
    
    ErrorTermObservationBST::ErrorTermObservationBST(aslam::backend::VectorExpression<1> robotPos, boost::shared_ptr<aslam::backend::Scalar> wallPoint, double y, double sigma2_n) :
    		//_observationErrorTerm(aslam::backend::Scalar(y).toExpression() - (aslam::backend::Scalar(1.0).toExpression() / (wallPoint->toExpression() - robotPos.toScalarExpression())))
    		_observationErrorTerm(ScalarExpression(boost::shared_ptr<ScalarExpressionNode>(new ScalarExpressionNodeConstant(y))) - (wallPoint->toExpression() / robotPos.toScalarExpression()))
    {
        Eigen::Matrix<double,1,1> invR;
        invR(0,0) = (1.0/sigma2_n);
      // Fill in the inverse covariance. In this scalar case, this is just an inverse variance.
        setInvR( invR );

        // create error term observation
        //_observationErrorTerm = aslam::backend::Scalar(y).toExpression();// - 1 / (wallPoint->toExpression() - robotPos);

      // Tell the super class about the design variables:
        JacobianContainer::set_t dvs;
        _observationErrorTerm.getDesignVariables(dvs);
        ErrorTermFs<1>::setDesignVariablesIterator(dvs.begin(), dvs.end());
    }

    ErrorTermObservationBST::~ErrorTermObservationBST()
    {

    }


    /// \brief evaluate the error term and return the weighted squared error e^T invR e
    double ErrorTermObservationBST::evaluateErrorImplementation()
    {
      // Build the error from the measurement _y and the design variables
    	error_t error;
        error(0) = _observationErrorTerm.toScalar();
//        std::cout << "The observation error is: " << std::endl << error(0) << std::endl;
        setError(error);
        return evaluateChiSquaredError();
    }


    /// \brief evaluate the jacobians
    void ErrorTermObservationBST::evaluateJacobiansImplementation(JacobianContainer & outJacobians)
    {
    	_observationErrorTerm.evaluateJacobians(outJacobians);
    }


  } // namespace backend
} // namespace aslam
