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

    // PTF: This needs comments
    //      Also, isn't the dimension implicit in R? (also d, also from the design variables)
    //      Also, doesn't dimensionErrorTerm == dimensionDesignVariables == R.cols() == R.rows() == d.size()?
    //      Note again that documentation is important because the order of the DVs in the
    //      vector is important. Also, the comments should point toward the factory function
  MarginalizationPriorErrorTerm(const std::vector<aslam::backend::DesignVariable*>& designVariables,
     const Eigen::VectorXd& d, const Eigen::MatrixXd& R, int dimensionErrorTerm, int dimensionDesignVariables);
  virtual ~MarginalizationPriorErrorTerm();

    // PTF: comments
  void removeTopDesignVariable();

    // PTF: comments
  bool isValid() { return _valid; }
  int numDesignVariables() { return _designVariables.size(); }
  aslam::backend::DesignVariable* getDesignVariable(int i);

private:
  MarginalizationPriorErrorTerm();

  virtual double evaluateErrorImplementation();
  virtual void evaluateJacobiansImplementation();

    // PTF: comments
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
