/*
 * LeapFrog.cpp
 *
 *  Created on: 04.08.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#include <aslam/backend/LeapFrog.hpp>
#include <aslam/backend/util/utils.hpp>
#include <aslam/Exceptions.hpp>

#include <sm/logging.hpp>

namespace aslam {
namespace backend {
namespace leapfrog {

bool simulate(const boost::shared_ptr<CostFunctionInterface>& potEnergy,
              RowVectorType& gradient,
              ColumnVectorType& momentum,
              const std::size_t numSteps,
              const double stepLength)
{
  SM_ASSERT_EQ(aslam::InvalidArgumentException, gradient.size(), momentum.size(), "");
  SM_ASSERT_POSITIVE(aslam::InvalidArgumentException, stepLength, "");

  ColumnVectorType dpos;
  const double stepLengthHalf = stepLength/2.; // precomputation

  // first half step of momentum
  momentum -= stepLengthHalf * gradient;

  // first full step for position
  dpos = stepLength * momentum;
  utils::applyStateUpdate(potEnergy->getDesignVariables(), dpos);

  SM_ALL_STREAM_NAMED("leapfrog", "Step 0 -- Momentum: " << momentum.transpose() << ", position update: " << dpos.transpose());

  // numSteps-1 full steps
  for(size_t l = 1; l < numSteps; ++l)
  {
    try
    {
      // momentum
      Timer timer("LeapFrog: Compute---Gradient", false);
      potEnergy->computeGradient(gradient);
      timer.stop();

      momentum -= stepLength * gradient;

      // position/sample
      dpos = stepLength * momentum;
      if(UNLIKELY(!dpos.allFinite())) { // we can abort the trajectory generation if it diverged
        return false;
      }
      utils::applyStateUpdate(potEnergy->getDesignVariables(), dpos);

      SM_ALL_STREAM_NAMED("leapfrog", "Step " << l << " -- Momentum: " << momentum.transpose() << ", position update: " << dpos.transpose());
    }
    catch (const std::exception& e)
    {
      SM_WARN_STREAM(e.what() << ": Compute gradient failed, terminating leap-frog simulation and rejecting sample");
      return false;
    }
  }

  // last half step for momentum
  try
  {
    Timer timer("LeapFrog: Compute---Gradient", false);
    potEnergy->computeGradient(gradient);
    timer.stop();
    momentum -= stepLengthHalf * gradient;
  }
  catch (const std::exception& e)
  {
    SM_WARN_STREAM(e.what() << ": Compute gradient failed, terminating leap-frog simulation and rejecting sample");
  }

  return true;
}

} /*namespace leapfrog */
} /* namespace backend */
} /* namespace aslam */
