#include <aslam/backend/ProbDataAssocPolicy.hpp>

#include <vector>

namespace aslam {
namespace backend {
ProbDataAssocPolicy::ProbDataAssocPolicy(ErrorTermGroups error_terms,
                                         double lambda) {
  error_terms_ = error_terms;
  scaling_factor_ = -lambda / 2;
}

void ProbDataAssocPolicy::callback() {
  for (ErrorTermGroup vect : *error_terms_) {
    double norm_factor = 0;
    std::vector<double> weights;
    weights.reserve(vect->size());
    for (ErrorTermPtr error_term : *vect) {
      // Update weights
      double new_weight =
          exp(scaling_factor_ * (error_term->getRawSquaredError()));
      norm_factor += new_weight;
      weights.push_back(new_weight);
    }

    for (std::size_t i = 0; i < vect->size(); i++) {
      boost::shared_ptr<FixedWeightMEstimator> m_estimator(
          (*vect)[i]->getMEstimatorPolicy<FixedWeightMEstimator>());
      assert(m_estimator);
      m_estimator->setWeight(weights[i] / norm_factor);
    }
  }
}
}  // namespace backend
}  // namespace aslam
