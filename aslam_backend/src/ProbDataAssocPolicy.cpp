#include <aslam/backend/ProbDataAssocPolicy.hpp>

#include <math.h>

#include <limits>
#include <vector>

namespace aslam {
namespace backend {
ProbDataAssocPolicy::ProbDataAssocPolicy(ErrorTermGroups error_terms, double v,
                                         int dimension)
    : error_terms_(error_terms), dimension_(dimension) {
  SM_ASSERT_TRUE(Exception, dimension > 0,
                 "The dimension of the error terms must be greater than zero");
  SM_ASSERT_TRUE(Exception, v > 0.0,
                 "The dof of the t-distribution must be greater than zero");
  if (v < std::numeric_limits<double>::infinity()) {
    v_ = v;
    t_exponent_ = -(v + dimension_) / 2.0;
    is_normal_ = false;
  } else {
    is_normal_ = true;
  }
}

void ProbDataAssocPolicy::callback() {
  for (ErrorTermGroup vect : *error_terms_) {
    double max_log_prob = - std::numeric_limits<double>::infinity();;
    std::vector<double> log_probs;
    std::vector<double> expected_weights;
    log_probs.reserve(vect->size());
    expected_weights.reserve(vect->size());
    for (ErrorTermPtr error_term : *vect) {
      // Update log_probs
      const double squared_error = error_term->getRawSquaredError();
      double log_prob;
      if (is_normal_) {
        log_prob = -squared_error / 2;
      } else {
        log_prob = (t_exponent_) * std::log1p(squared_error / v_);
        const double expected_prob = (v_ + dimension_) / (v_ + squared_error);
        expected_weights.push_back(expected_prob);
      }

      if (log_prob > max_log_prob) {
        max_log_prob = log_prob;
      }
      log_probs.push_back(log_prob);
    }
    double log_norm_constant = 0;
    for (double log_p : log_probs) {
      log_norm_constant += std::exp(log_p - max_log_prob);
    }
    log_norm_constant = std::log(log_norm_constant) + max_log_prob;

    for (std::size_t i = 0; i < vect->size(); i++) {
      boost::shared_ptr<FixedWeightMEstimator> m_estimator(
          (*vect)[i]->getMEstimatorPolicy<FixedWeightMEstimator>());
      SM_ASSERT_TRUE(Exception, m_estimator,
                     "The error term does not have a FixedWeightMEstimator");
      if (is_normal_) {
        m_estimator->setWeight(std::exp(log_probs[i] - log_norm_constant));
      } else {
        m_estimator->setWeight(std::exp(log_probs[i] - log_norm_constant) *
                               expected_weights[i]);
      }
    }
  }
}
}  // namespace backend
}  // namespace aslam
