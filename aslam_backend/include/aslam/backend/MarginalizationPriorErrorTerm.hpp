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

  MarginalizationPriorErrorTerm(const std::vector<aslam::backend::DesignVariable*>& designVariables,
     const Eigen::VectorXd& d, const Eigen::MatrixXd& R, int dimensionErrorTerm, int dimensionDesignVariables);
  virtual ~MarginalizationPriorErrorTerm();

  void removeTopDesignVariable();

  bool isValid() { return _valid; }
  int numDesignVariables() { return _designVariables.size(); }
  aslam::backend::DesignVariable* getDesignVariable(int i);

private:
  MarginalizationPriorErrorTerm();

  virtual double evaluateErrorImplementation();
  virtual void evaluateJacobiansImplementation();

  Eigen::VectorXd getDifferenceSinceMarginalization();

  bool _valid;
  int _dimensionDesignVariables;
  std::vector<aslam::backend::DesignVariable*> _designVariables;
  Eigen::VectorXd _d;
  Eigen::MatrixXd _R; // R from the QR decomposition!!!
  Eigen::MatrixXd _M;
  // store values of design variables at time of marginalization
  std::vector<Eigen::MatrixXd> _designVariableValuesAtMarginalization;
};

} /* namespace backend */
} /* namespace aslam */
#endif /* MARGINALIZATIONPRIORERRORTERM_HPP_ */
