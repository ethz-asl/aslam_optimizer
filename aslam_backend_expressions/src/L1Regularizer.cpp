/*
 * L1Regularizer.cpp
 *
 *  Created on: 22.09.2015
 *      Author: Ulrich Schwesinger
 */

#include <aslam/backend/L1Regularizer.hpp>

namespace aslam {
namespace backend {

using namespace std;

L1Regularizer::L1Regularizer(const vector<Scalar*>& dvs, const double beta) :
    _dvs(dvs) {

  vector<DesignVariable*> dvstmp(dvs.begin(), dvs.end());
  setDesignVariables(dvstmp);
  setWeight(beta);
}

/// \brief evaluate the error term and return the scalar error \f$ e \f$
double L1Regularizer::evaluateErrorImplementation() {

  double sum = 0.;
  for (auto& dv : _dvs) {
    if (dv->isActive())
      sum += fabs(dv->getParameters()(0,0));
  }
  return sum;
}

/// \brief evaluate the Jacobians for \f$ e \f$
void L1Regularizer::evaluateJacobiansImplementation(JacobianContainer & outJacobians) {
  Eigen::MatrixXd J(1,1);
  for (auto& dv : _dvs) {
    if (dv->isActive()) {
      Eigen::MatrixXd p = dv->getParameters();
      J(0,0) = (0.0 < p(0,0)) - (p(0,0) < 0.0); // branchless signum version
      outJacobians.add(dv, J);
    }
  }
}

}
}
