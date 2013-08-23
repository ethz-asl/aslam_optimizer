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
#include <aslam/backend/DenseMatrix.hpp>

#include <iostream>

namespace aslam {
namespace backend {

void marginalize(
			std::vector<aslam::backend::DesignVariable*>& inDesignVariables,
			std::vector<aslam::backend::ErrorTerm*>& inErrorTerms,
			int numberOfInputDesignVariablesToRemove,
			bool useMEstimator,
			boost::shared_ptr<aslam::backend::MarginalizationPriorErrorTerm>& outPriorErrorTermPtr)
{
          // Partition the design varibles into removed/remaining.
		  int dimOfDesignVariablesToRemove = 0;
		  std::vector<aslam::backend::DesignVariable*> remainingDesignVariables;
		  int k = 0;
		  for(std::vector<aslam::backend::DesignVariable*>::const_iterator it = inDesignVariables.begin(); it != inDesignVariables.end(); ++it)
		  {
			  if (k < numberOfInputDesignVariablesToRemove)
			  {
				  dimOfDesignVariablesToRemove += (*it)->minimalDimensions();
			  } else
			  {
				  remainingDesignVariables.push_back(*it);
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

			int dim = 0;
			std::vector<size_t> originalRowBase;
			for(std::vector<aslam::backend::ErrorTerm*>::iterator it = inErrorTerms.begin(); it != inErrorTerms.end(); ++it)
			{
				originalRowBase.push_back((*it)->rowBase());
				(*it)->setRowBase(dim);
				dim += (*it)->dimension();
			}


          
		  aslam::backend::DenseQrLinearSystemSolver qrSolver;

          qrSolver.initMatrixStructure(inDesignVariables, inErrorTerms, false);

		  //std::cout << "Marginalization optimization problem initialized with " << inDesignVariables.size() << " design variables and " << inErrorTerms.size() << " error terms" << std::endl;
		  //std::cout << "The Jacobian matrix is " << dim << " x " << columnBase << std::endl;

		  qrSolver.evaluateError(1, useMEstimator);
		  qrSolver.buildSystem(1, useMEstimator);


		  const Eigen::MatrixXd& jacobian = qrSolver.getJacobian();
		  //Eigen::MatrixXd jacobian = qrSolver.Jacobian()->toDense();
		  const Eigen::VectorXd& b = qrSolver.e();

		  // check dimension of jacobian
		  int jrows = jacobian.rows();
		  int jcols = jacobian.cols();

		  int dimOfRemainingDesignVariables = jcols - dimOfDesignVariablesToRemove;
		  int dimOfPriorErrorTerm = jrows;

		  Eigen::MatrixXd R_reduced;
		  Eigen::VectorXd d_reduced;
		  if (jrows < jcols)
		  {
			  // overdetermined LSE, don't do QR
			  R_reduced = jacobian;
			  d_reduced = b;
		  } else {
			  // PTF: Do we know what will happen when the jacobian matrix is rank deficient?

              // do QR decomposition
              Eigen::HouseholderQR<Eigen::MatrixXd> qr(jacobian);
			  Eigen::MatrixXd Q = qr.householderQ();
			  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
			  Eigen::VectorXd d = Q.transpose()*b;

			  //std::cout << "R matrix is: " << std::endl << R << std::endl;

              // cut off the zero rows at the bottom
              R_reduced = R.block(dimOfDesignVariablesToRemove, dimOfDesignVariablesToRemove, dimOfRemainingDesignVariables, dimOfRemainingDesignVariables);

              d_reduced = d.segment(dimOfDesignVariablesToRemove, dimOfRemainingDesignVariables);
              dimOfPriorErrorTerm = dimOfRemainingDesignVariables;
		  }


//		  std::cout << "R reduced matrix is: " << std::endl << R_reduced << std::endl;
//		  std::cout << "d reduced" << std::endl << d_reduced << std::endl;
//		  std::cout << "The jacobian of the marginalization is: " << std::endl << jacobian << std::endl;

		  // now create the new error term
		  boost::shared_ptr<aslam::backend::MarginalizationPriorErrorTerm> err(new aslam::backend::MarginalizationPriorErrorTerm(remainingDesignVariables, d_reduced, R_reduced));
		  //std::cout << "address of new prior error term: " << err.get() << std::endl;

		  outPriorErrorTermPtr.swap(err);
		  //std::cout << "address of new prior error term: " << outPriorErrorTermPtr.get() << std::endl;

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


} /* namespace backend */
} /* namespace aslam */
