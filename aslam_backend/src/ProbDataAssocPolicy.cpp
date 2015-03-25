#include <aslam/backend/ProbDataAssocPolicy.hpp>

#include <math.h>

#include <limits>
#include <vector>

namespace aslam {
namespace backend {
ProbDataAssocPolicy::ProbDataAssocPolicy(ErrorTermGroups error_terms, double v,
                                         int dimension) {
  error_terms_ = error_terms;
  dimension_ = dimension;
  if (v < std::numeric_limits<double>::infinity()) {
    v_ = v;
    t_exponent_ = -(v + dimension_) / 2.0;
    use_gaussian = false;
  } else {
    use_gaussian = true;
  }
}

void ProbDataAssocPolicy::callback() {
  for (ErrorTermGroup vect : *error_terms_) {
    bool init_max = false;
    double max_log_prob = 0;
    std::vector<double> log_probs;
    std::vector<double> expected_probs;
    log_probs.reserve(vect->size());
    expected_probs.reserve(vect->size());
    for (ErrorTermPtr error_term : *vect) {
      // Update log_probs
      double squared_error = error_term->getRawSquaredError();
      double log_prob;
      if (use_gaussian) {
        log_prob = -squared_error / 2;
      } else {
        log_prob = (t_exponent_) * std::log1p(squared_error / v_);
        double expected_prob = (v_ + dimension_) / (v_ + squared_error);
        expected_probs.push_back(expected_prob);
      }
      if (!init_max) {
        max_log_prob = log_prob;
        init_max = true;
      } else if (log_prob > max_log_prob) {
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
      assert(m_estimator);
      if (use_gaussian) {
        m_estimator->setWeight(std::exp(log_probs[i] - log_norm_constant));
      } else {
        m_estimator->setWeight(std::exp(log_probs[i] - log_norm_constant) *
                               expected_probs[i]);
      }
    }
  }
}
}  // namespace backend
}  // namespace aslam
