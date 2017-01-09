#include <aslam/backend/LineSearchTrustRegionPolicy.hpp>
#include <aslam/backend/util/CommonDefinitions.hpp>
#include <sm/assert_macros.hpp>

namespace aslam {
namespace backend {

LineSearchTrustRegionPolicy::LineSearchTrustRegionPolicy(double scaleStep, bool resetScaleAfterSuccess) : _scaleStep(scaleStep), _resetScaleAfterSuccess(resetScaleAfterSuccess) {
  SM_ASSERT_GE_LT(std::runtime_error, _scaleStep, 0.0, 1.0, "The scale must be on the interval (0,1).");
}
LineSearchTrustRegionPolicy::~LineSearchTrustRegionPolicy() {}


/// \brief called by the optimizer when an optimization is starting
void LineSearchTrustRegionPolicy::optimizationStartingImplementation(double /* J */) {

}

// Returns true if the solution was successful
bool LineSearchTrustRegionPolicy::solveSystemImplementation(double /* J */, bool previousIterationFailed, int nThreads, Eigen::VectorXd& outDx) {
  bool success = true;
  if(isFirstIteration() || !previousIterationFailed) {
    Timer timeBuild("LsGnTrustRegionPolicy: Build linear system", false);
    _solver->buildSystem(nThreads, true);
    timeBuild.stop();
    Timer timeSolve("LsGnTrustRegionPolicy: Solve linear system", false);
    success = _solver->solveSystem(outDx);
    timeSolve.stop();
    if(isFirstIteration() || _resetScaleAfterSuccess){
      _currentScale = 1.0;
    } else {
      _currentScale = std::min(1.0, _currentScale / _scaleStep);
    }
  } else {
    _currentScale *= _scaleStep;
    outDx*= _scaleStep;
  }

  return success;
}

/// \brief print the current state to a stream (no newlines).
std::ostream & LineSearchTrustRegionPolicy::printState(std::ostream & out) const {
  out << "LS (" << _currentScale << ")" << std::endl;
  return out;
}


void LineSearchTrustRegionPolicy::setScaleStep(double scale){
  SM_ASSERT_GE_LT(std::runtime_error, scale, 0.0, 1.0, "The scale must be on the interval (0,1).");
  _scaleStep = scale;
}

bool LineSearchTrustRegionPolicy::requiresAugmentedDiagonal() const {
  return false;
}

} // namespace backend
} // namespace aslam
