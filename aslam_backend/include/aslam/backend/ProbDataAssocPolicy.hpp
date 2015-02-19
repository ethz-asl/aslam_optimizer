#ifndef ASLAM_PROB_DATA_ASSOC_POLICY_HPP
#define ASLAM_PROB_DATA_ASSOC_POLICY_HPP

#include <math.h>

#include <boost/shared_ptr.hpp>

#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/PerIterationCallback.hpp>

#include <vector>

namespace aslam {
namespace backend {

// Update the weights of the error terms using a gaussian of the squared error,
// i.e: exp(-lambda/2*(y -f(x))^2)
// The matrix of error terms contains, in each row, the error terms whose
// weights must be normalized together
class ProbDataAssocPolicy : public PerIterationCallback {
 private:
  typedef boost::shared_ptr<ErrorTerm> ErrorTerm_Ptr;
  typedef boost::shared_ptr<std::vector<ErrorTerm_Ptr>>
      ErrorTermNormalizationGroup_Ptr;
  typedef boost::shared_ptr<std::vector<ErrorTermNormalizationGroup_Ptr>>
      ErrorTermGroups_Ptr;
  ErrorTermGroups_Ptr error_terms_;
  double scaling_factor_;

 public:
  ProbDataAssocPolicy(ErrorTermGroups_Ptr error_terms, double lambda);

  // The optimizer will call this function before evaluating the error terms
  void callback();
};
}  // namespace backend
}  // namespace aslam

#endif /*ASLAM_PROB_DATA_ASSOC_POLICY_HPP*/
