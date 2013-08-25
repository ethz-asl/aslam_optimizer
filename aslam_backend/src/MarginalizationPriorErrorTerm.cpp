/*
 * MarginalizationPriorErrorTerm.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: mbuerki
 */

#include <aslam/backend/MarginalizationPriorErrorTerm.hpp>
#include <Eigen/Dense>

namespace aslam {
namespace backend {

MarginalizationPriorErrorTerm::MarginalizationPriorErrorTerm(const std::vector<aslam::backend::DesignVariable*>& designVariables,
    const Eigen::VectorXd& d, const Eigen::MatrixXd& R)
: aslam::backend::ErrorTermDs(R.rows()), _designVariables(designVariables), _d(d), _R(R), _dimensionDesignVariables(R.cols())
{

  //_valid = true;
  // PTF: Hrm...here is a big weakness of the current optimizer code. We should have
  //      different base classes for different uncertainty types (scalar, diagonal, matrix, none)
  //      to avoid big matrix multiplications during optimization.
  setInvR(Eigen::MatrixXd::Identity(R.cols(), R.cols()));

  //_M = Eigen::MatrixXd::Identity(R.cols(), R.cols());

  // set all design variables
  for(vector<aslam::backend::DesignVariable*>::iterator it = _designVariables.begin(); it != _designVariables.end(); ++it)
  {
    Eigen::MatrixXd values;
    (*it)->getParameters(values);
    // PTF: Nice variable name.
    _designVariableValuesAtMarginalization.push_back(values);
  }
  setDesignVariables(designVariables);

}

MarginalizationPriorErrorTerm::~MarginalizationPriorErrorTerm() {
  // TODO Auto-generated destructor stub
}

double MarginalizationPriorErrorTerm::evaluateErrorImplementation()
{
  Eigen::VectorXd diff = getDifferenceSinceMarginalization();
  // ASSERT DIMENSIONS MATCH
  std::cout << "Diff looks like this: " << std::endl << diff << std::endl;
  std::cout << "R looks like this: " << std::endl << _R << std::endl;
  std::cout << "d looks like this: " << std::endl << _d << std::endl;
  // PTF: Why a negative here?
  SM_ASSERT_EQ(aslam::Exception, diff.rows(), _R.cols(), "Dimension of R and the minimal difference vector mismatch!");
  SM_ASSERT_EQ(aslam::Exception, _d.rows(), _R.rows(), "Dimension of R and the d mismatch!");
  Eigen::VectorXd currentError = -(_d - _R*diff);
  std::cout << "current error looks like this: " << std::endl << currentError << std::endl;
  setError(currentError);
  return evaluateChiSquaredError();

}

// Computes the difference vector of all design variables between the linearization point at marginalization and the current guess, on the tangent space (i.e. log(x_bar - x))
Eigen::VectorXd MarginalizationPriorErrorTerm::getDifferenceSinceMarginalization()
{
  Eigen::VectorXd diff = Eigen::VectorXd(_dimensionDesignVariables);
  // ASSERT designvariables == designvariablesatmarginalization (i.e. dimensions, number, etc.)
  int index = 0;
  std::vector<Eigen::MatrixXd>::iterator it_marg = _designVariableValuesAtMarginalization.begin();
  std::vector<aslam::backend::DesignVariable*>::iterator it_current = _designVariables.begin();
  for(;it_current != _designVariables.end(); ++it_current, ++it_marg)
  {
	  // retrieve current value (xbar) and value at marginalization(xHat)
      Eigen::MatrixXd xHat = *it_marg;
      
      std::cout << "xhat:" << std::endl << xHat << std::endl;
      
      //get minimal difference in tangent space
      Eigen::VectorXd diffVector;
      //Eigen::MatrixXd Mblock;
      (*it_current)->minimalDifference(xHat, diffVector);
      std::cout << "diffVector:" << std::endl << diffVector << std::endl;
      int base = index;
      int dim = diffVector.rows();
      diff.segment(index, dim) = diffVector;
      index += dim;

//      SM_ASSERT_TRUE(aslam::Exception, (Mblock.rows()==dim)&&(Mblock.cols()==dim), "Dimension mismatch!");
//      _M.block(base, base, dim, dim) = Mblock;
  }
  std::cout << "diff is: " << std::endl << diff << std::endl;
  return diff;
}

void MarginalizationPriorErrorTerm::evaluateJacobiansImplementation(JacobianContainer & outJ)
{
  int colIndex = 0;
  std::vector<Eigen::MatrixXd>::iterator it_marg = _designVariableValuesAtMarginalization.begin();
  for(vector<aslam::backend::DesignVariable*>::iterator it = _designVariables.begin(); it != _designVariables.end(); ++it, ++it_marg)
  {
    int dimDesignVariable = (*it)->minimalDimensions();
    Eigen::MatrixXd M;
    Eigen::VectorXd diff;
    (*it)->minimalDifferenceAndJacobian(*it_marg, diff, M);
    SM_ASSERT_EQ(aslam::Exception, M.rows(), dimDesignVariable, "Minimal difference jacobian and design variable dimension mismatch!");
    outJ.add(*it, _R.block(0, colIndex, _dimensionErrorTerm, dimDesignVariable)*M);
    colIndex += dimDesignVariable;
  }

}

aslam::backend::DesignVariable* MarginalizationPriorErrorTerm::getDesignVariable(int i)
{
  return _designVariables[i];
}

//// removes the top/first design variable (that is, the design variable a the beginning of the design variable vector) from the error term
//void MarginalizationPriorErrorTerm::removeTopDesignVariable()
//{
//	SM_ASSERT_TRUE(aslam::Exception, _designVariables.size() > 0, "Cannot remove top design variable because there are none left!");
//	// remove first design variable
//	aslam::backend::DesignVariable* dv = _designVariables[0];
//	int dv_dim = dv->minimalDimensions();
//	_designVariables.erase(_designVariables.begin());
//	_designVariableValuesAtMarginalization.erase(_designVariableValuesAtMarginalization.begin());
//
//	int newStartRow = 0;
//
//	if (_dimensionErrorTerm >= _dimensionDesignVariables)
//	{
//		_dimensionErrorTerm -= dv_dim;
//		newStartRow = dv_dim;
//	}
//	_dimensionDesignVariables -= dv_dim;
//
//
////	Eigen::MatrixXd Rreduced = _R.block(dv_dim, dv_dim, _dimensionErrorTerm, _dimensionErrorTerm); // maybe -1?
////	Eigen::MatrixXd Mreduced = _M.block(dv_dim, dv_dim, _dimensionErrorTerm, _dimensionErrorTerm);
//	_R = _R.block(newStartRow, dv_dim, _dimensionErrorTerm, _dimensionErrorTerm); // maybe -1?
//
////	Eigen::VectorXd dreduced = _d.block(dv_dim, 0, _dimensionErrorTerm, 1);
//	_d = _d.segment(newStartRow, _dimensionErrorTerm);
//
//	setInvR(Eigen::MatrixXd::Identity(_dimensionErrorTerm, _dimensionErrorTerm));
//
//	resizeJacobianContainer(_dimensionErrorTerm);
//}



} /* namespace backend */
} /* namespace aslam */
