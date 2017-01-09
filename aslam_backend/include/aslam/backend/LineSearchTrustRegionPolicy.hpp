#ifndef ASLAM_BACKEND_LINE_SEARCH_TRUST_REGION_POLICY_HPP
#define ASLAM_BACKEND_LINE_SEARCH_TRUST_REGION_POLICY_HPP

#include <aslam/backend/TrustRegionPolicy.hpp>

namespace aslam {
namespace backend {

class LineSearchTrustRegionPolicy : public TrustRegionPolicy {
 public:
  /// \brief Construct LineSearchTrustRegionPolicy
  /// @param scaleStep: How much to multiply the current scale with when an iteration fails or divide by if it succeeds unless resetScaleAfterSuccess.
  /// @param resetScaleAfterSuccess: @see scaleStep
  LineSearchTrustRegionPolicy(double scaleStep = 0.5, bool resetScaleAfterSuccess = false);
  virtual ~LineSearchTrustRegionPolicy();

  /// \brief called by the optimizer when an optimization is starting
  virtual void optimizationStartingImplementation(double J) override;

  /// \brief Returns true if the solution was successful
  virtual bool solveSystemImplementation(double J, bool previousIterationFailed, int nThreads, Eigen::VectorXd& outDx) override;

  /// \brief print the current state to a stream (no newlines).
  virtual std::ostream & printState(std::ostream & out) const override;

  virtual std::string name() const override { return "line_search"; }
  virtual bool requiresAugmentedDiagonal() const override;

  double getScaleStep() { return _scaleStep; }
  void setScaleStep(double scale);

  bool isResetScaleAfterSuccess() const { return _resetScaleAfterSuccess; }
  void setResetScaleAfterSuccess(bool resetScaleAfterSuccess) { _resetScaleAfterSuccess = resetScaleAfterSuccess; }
 private:
  double _scaleStep;
  bool _resetScaleAfterSuccess;
  double _currentScale = 1.0;
};

}  // namespace backend
}  // namespace aslam

#endif /* ASLAM_BACKEND_GAUSS_NEWTON_TRUST_REGION_POLICY_HPP */
