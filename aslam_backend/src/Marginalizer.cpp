/*
 * Marginalizer.cpp
 *
 *  Created on: May 24, 2013
 *      Author: mbuerki
 */

#include "aslam/backend/Marginalizer.hpp"

#include <aslam/backend/DenseQrLinearSystemSolver.hpp>
#include <Eigen/QR>
#include <Eigen/Dense>

#include <iostream>

namespace aslam {
namespace backend {

Marginalizer::Marginalizer() {
	// TODO Auto-generated constructor stub

}

Marginalizer::~Marginalizer() {
	// TODO Auto-generated destructor stub
}

void Marginalizer::operator () (
		std::vector<aslam::backend::DesignVariable*>& inDesignVariables,
		std::vector<aslam::backend::ErrorTerm*>& inErrorTerms,
		int numberOfInputDesignVariablesToRemove,
		aslam::backend::MarginalizationPriorErrorTerm** outPriorErrorTerm) const
{

		  std::vector<aslam::backend::DesignVariable*> remainingDesignVariables;
		  int k = 0;
		  //int dimOfDesignVariablesToRemove = 0;
		  for(std::vector<aslam::backend::DesignVariable*>::const_iterator it = inDesignVariables.begin(); it != inDesignVariables.end(); ++it)
		  {
			  if (k >= numberOfInputDesignVariablesToRemove)
			  {
				  remainingDesignVariables.push_back(*it);
				  //dimOfDesignVariablesToRemove += (*it)->minimalDimensions();
			  }
			  k++;
		  }

		  // store original block indices to prevent side effects
		  std::vector<int> originalBlockIndices;
		  std::vector<int> originalColumnBase;
			// assign block indices
			int columnBase = 0;
			for (size_t i = 0; i < inDesignVariables.size(); ++i) {
				originalBlockIndices.push_back(inDesignVariables[i]->blockIndex());
				originalColumnBase.push_back(inDesignVariables[i]->columnBase());

				inDesignVariables[i]->setBlockIndex(i);
				inDesignVariables[i]->setColumnBase(columnBase);
			  columnBase += inDesignVariables[i]->minimalDimensions();
			}

			// THIS CHANGES THE STATE OF FOREIGN OBJECTS!!!??? (i.e. it's causing side effects)
			int dim = 0;
			std::vector<size_t> originalRowBase;
			for(std::vector<aslam::backend::ErrorTerm*>::iterator it = inErrorTerms.begin(); it != inErrorTerms.end(); ++it)
			{
				originalRowBase.push_back((*it)->rowBase());
				(*it)->setRowBase(dim);
				dim += (*it)->dimension();
			}


		  aslam::backend::DenseQrLinearSystemSolver qrSolver;
		  qrSolver.initMatrixStructure(inDesignVariables, inErrorTerms, true);

		  std::cout << "Marginalization optimization problem initialized with " << inDesignVariables.size() << " design variables and " << inErrorTerms.size() << " error terms" << std::endl;
		  std::cout << "The Jacobian matrix is " << dim << " x " << columnBase << std::endl;

		  qrSolver.evaluateError(1, true);
		  qrSolver.buildSystem(1, true);

		  Eigen::MatrixXd jacobian = qrSolver.Jacobian()->toDense();
		  const Eigen::VectorXd b = qrSolver.e();

		  Eigen::HouseholderQR<Eigen::MatrixXd> qr(jacobian);
		  Eigen::MatrixXd Q = qr.householderQ();
		  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

		  // make r square
		  int rrows = R.rows();
		  int rcols = R.cols();
		  // HANDLE CASES WHERE THERE ARE FEWER ERROR TERMS THAN DESIGN VARIABLES!!!
		  Eigen::MatrixXd R_reduced = R;
		  Eigen::VectorXd d = Q.transpose()*b;
		  Eigen::VectorXd d_reduced = d;

		  int numberOfColumnsToKeep = rcols - numberOfInputDesignVariablesToRemove;
		  int dimOfRemainingDesignVariables = rcols;
		  int dimOfPriorErrorTerm = rrows;

		  int firstColumn;
		  if (rrows >= rcols)
		  {
		    // cut off the zero rows at the bottom
			R_reduced = R.block(0, numberOfInputDesignVariablesToRemove, rcols, numberOfColumnsToKeep);
			d_reduced = d.block(numberOfInputDesignVariablesToRemove,0, numberOfColumnsToKeep, 1);
			dimOfRemainingDesignVariables = numberOfColumnsToKeep;
			dimOfPriorErrorTerm = numberOfColumnsToKeep;
		  } else
		  {
			// underdetermined LSE
			// TODO: figure out how to handle this
		  }

		  std::cout << "R matrix is: " << R << std::endl;
		  std::cout << "R reduced matrix is: " << R_reduced << std::endl;
		  std::cout << "The determinant of the R matrix is: " << R_reduced.determinant() << std::endl;
		  std::cout << "The jacobian of the marginalization is: " << std::endl << jacobian << std::endl;

		  // now create the new error term
		  *outPriorErrorTerm = new aslam::backend::MarginalizationPriorErrorTerm(remainingDesignVariables, d_reduced, R_reduced, dimOfPriorErrorTerm, dimOfRemainingDesignVariables);

		  int n = (*outPriorErrorTerm)->numDesignVariables();
		  // set output variables
		  //*outPriorErrorTermPtrPtr = newErrorTerm;

		  // restore initial block indices to prevent side effects
		for (size_t i = 0; i < inDesignVariables.size(); ++i) {
			inDesignVariables[i]->setBlockIndex(originalBlockIndices[i]);
			inDesignVariables[i]->setColumnBase(originalColumnBase[i]);
		}
		int index = 0;
		for(std::vector<aslam::backend::ErrorTerm*>::iterator it = inErrorTerms.begin(); it != inErrorTerms.end(); ++it)
		{
			(*it)->setRowBase(originalRowBase[index++]);
		}
}


}

 /* namespace backend */
} /* namespace aslam */
