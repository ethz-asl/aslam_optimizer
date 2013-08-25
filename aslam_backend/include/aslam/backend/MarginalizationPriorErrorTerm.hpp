/*
 * FLSPriorErrorTerm.hpp
 *
 *  Created on: Apr 26, 2013
 *      Author: mbuerki
 */

#ifndef MARGINALIZATIONPRIORERRORTERM_HPP_
#define MARGINALIZATIONPRIORERRORTERM_HPP_

#include <aslam/backend/ErrorTermDs.hpp>
#include <aslam/backend/DesignVariable.hpp>

#include <boost/shared_ptr.hpp>

#include <list>

namespace aslam {
namespace backend {

class MarginalizationPriorErrorTerm : public aslam::backend::ErrorTermDs {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef boost::shared_ptr<aslam::backend::MarginalizationPriorErrorTerm> Ptr;

  // creates the marginalization error term of the form e(x) = d - R*x
  // designVariables: 	the design variables of this marginalization prior error term
  MarginalizationPriorErrorTerm(const std::vector<aslam::backend::DesignVariable*>& designVariables,
     const Eigen::VectorXd& d, const Eigen::MatrixXd& R);
  virtual ~MarginalizationPriorErrorTerm();

//  // removes the top/first design variable (that is, the design variable a the beginning of the design variable vector) from the error term
//  // currently only used for integrity testing. may be removed at some point
//  void removeTopDesignVariable();

  int numDesignVariables() { return _designVariables.size(); }
  aslam::backend::DesignVariable* getDesignVariable(int i);

private:
  MarginalizationPriorErrorTerm();

  virtual double evaluateErrorImplementation();
  virtual void evaluateJacobiansImplementation(JacobianContainer & outJ);

  // computes the minimal difference of all design variables between the linearization point at marginalization and the current guess (i.e. log(x_bar - x))
  Eigen::VectorXd getDifferenceSinceMarginalization();

  int _dimensionDesignVariables;
  std::vector<aslam::backend::DesignVariable*> _designVariables;
  Eigen::VectorXd _d;
  Eigen::MatrixXd _R; // R from the QR decomposition!!!
  //Eigen::MatrixXd _M;
  // store values of design variables at time of marginalization
  std::vector<Eigen::MatrixXd> _designVariableValuesAtMarginalization;
};

} /* namespace backend */
} /* namespace aslam */
#endif /* MARGINALIZATIONPRIORERRORTERM_HPP_ */
