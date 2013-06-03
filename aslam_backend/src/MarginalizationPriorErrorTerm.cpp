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
    const Eigen::VectorXd& d, const Eigen::MatrixXd& R, int dimensionErrorTerm, int dimensionDesignVariables)
: aslam::backend::ErrorTermDs(dimensionErrorTerm), _designVariables(designVariables), _d(d), _R(R), _dimensionDesignVariables(dimensionDesignVariables)
{

  _valid = true;
  setInvR(Eigen::MatrixXd::Identity(dimensionErrorTerm, dimensionErrorTerm));

  _M = Eigen::MatrixXd::Identity(dimensionDesignVariables, dimensionDesignVariables);

  // set all design variables
  for(vector<aslam::backend::DesignVariable*>::iterator it = _designVariables.begin(); it != _designVariables.end(); ++it)
  {
    Eigen::MatrixXd values;
    (*it)->getParameters(values);
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
  Eigen::VectorXd currentError = -(_d - _R*diff);
  std::cout << "current error looks like this: " << std::endl << currentError << std::endl;
  setError(currentError);
  return evaluateChiSquaredError();

}

Eigen::VectorXd MarginalizationPriorErrorTerm::getDifferenceSinceMarginalization()
{
  Eigen::VectorXd diff = Eigen::VectorXd(_dimensionDesignVariables);
  // ASSERT designvariables == designvariablesatmarginalization (i.e. dimensions, number, etc.)
  int index = 0;
  std::vector<Eigen::MatrixXd>::iterator it_marg = _designVariableValuesAtMarginalization.begin();
  std::vector<aslam::backend::DesignVariable*>::iterator it_current = _designVariables.begin();
  std::cout << "number of design variables: " << _designVariables.size() << " and number of design variable values at marginalization: " << _designVariableValuesAtMarginalization.size() << std::endl;
  for(;it_current != _designVariables.end(); ++it_current, ++it_marg)
  {
	  // retrieve current value (xar) and value at marginalization(xHat)
      Eigen::MatrixXd xHat = *it_marg;

      std::cout << "xhat:" << std::endl << xHat << std::endl;

      //get minimal difference in tangent space
      Eigen::VectorXd diffVector;
      Eigen::MatrixXd Mblock;
      (*it_current)->minimalDifferenceAndJacobian(xHat, diffVector, Mblock);
      std::cout << "diffVector:" << std::endl << diffVector << std::endl;
      std::cout << "Mblock:" << std::endl << Mblock << std::endl;
      int base = index;
      int dim = diffVector.rows();
      // TODO: do this as block copy
      for(int i = 0; i < diffVector.rows(); i++)
      {
        diff(index++) = diffVector(i);
      }
      SM_ASSERT_TRUE(aslam::Exception, (Mblock.rows()==dim)&&(Mblock.cols()==dim), "Dimension mismatch!");
      _M.block(base, base, dim, dim) = Mblock;
  }
  return diff;
}

void MarginalizationPriorErrorTerm::evaluateJacobiansImplementation()
{
  int index = 0;
  // only valid for vector spaces!
  for(vector<aslam::backend::DesignVariable*>::iterator it = _designVariables.begin(); it != _designVariables.end(); ++it)
  {
    int dim = (*it)->minimalDimensions();
    // MAKE THIS MORE EFFICIENT!!!
    Eigen::MatrixXd bl = _R;
    //std::cout << "block looks like this:" << std::endl << bl << std::endl;
    Eigen::MatrixXd E = bl.block(0, index++, _dimensionErrorTerm, dim);
    //std::cout << "block looks like this:" << std::endl << E << std::endl;
    _jacobians.add(*it, E);
  }

}

aslam::backend::DesignVariable* MarginalizationPriorErrorTerm::getDesignVariable(int i)
{
  return _designVariables[i];
}

void MarginalizationPriorErrorTerm::removeTopDesignVariable()
{
	// remove first design variable
	aslam::backend::DesignVariable* dv = _designVariables[0];
	int dv_dim = dv->minimalDimensions();
	_designVariables.erase(_designVariables.begin());
	_designVariableValuesAtMarginalization.erase(_designVariableValuesAtMarginalization.begin());
	// check if this does the right thing!
	if (_dimensionErrorTerm >= _dimensionDesignVariables)
	{
		_dimensionErrorTerm -= dv_dim;
	}
	_dimensionDesignVariables -= dv_dim;


	Eigen::MatrixXd Rreduced = _R.block(dv_dim, dv_dim, _dimensionErrorTerm, _dimensionErrorTerm); // maybe -1?
	Eigen::MatrixXd Mreduced = _M.block(dv_dim, dv_dim, _dimensionErrorTerm, _dimensionErrorTerm);
	_R = Rreduced;
	_M = Mreduced;
	Eigen::VectorXd dreduced = _d.block(dv_dim, 0, _dimensionErrorTerm, 1);
	_d = dreduced;

	setInvR(Eigen::MatrixXd::Identity(_dimensionErrorTerm, _dimensionErrorTerm));

	resizeJacobianContainer(_dimensionErrorTerm);

}



} /* namespace backend */
} /* namespace aslam */
