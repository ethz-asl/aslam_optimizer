#include <aslam/backend/KinematicChain.hpp>

namespace aslam {
namespace backend {

void aslam::backend::CoordinateFrame::initGlobalsWithoutParent() {
  R_G_L = R_P_L;
  pG = p;
  omegaG = omega;
  vG = v;
  alphaG = alpha;
  aG = a;
}

}  // namespace backend
}  // namespace aslam
