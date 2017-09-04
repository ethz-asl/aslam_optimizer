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

const aslam::backend::EuclideanExpression& CoordinateFrame::getVG() const {
  if (pp && vG.isEmpty()) {
    vG = pp->getVG() + pp->getR_G_L() * v + pp->getOmegaG().cross(pp->getR_G_L() * p);  //TODO have a caching variable for (pp->getR_G_L() * p)!
  }
  return vG;
}

const aslam::backend::EuclideanExpression& CoordinateFrame::getAG() const {
  if (pp && aG.isEmpty()) {
    aG = pp->getAG() + pp->getR_G_L() * a + pp->getOmegaG().cross(pp->getR_G_L() * v) + pp->getAlphaG().cross(pp->getR_G_L() * p) + pp->getOmegaG().cross(pp->getOmegaG().cross(pp->getR_G_L() * p));
  }
  return aG;
}

}  // namespace backend
}  // namespace aslam
