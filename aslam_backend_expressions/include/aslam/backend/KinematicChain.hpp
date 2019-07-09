/*
 * KinematicChain.hpp
 *
 *  Created on: Oct 7, 2014
 *      Author: hannes
 */

#ifndef KINEMATICCHAIN_HPP_
#define KINEMATICCHAIN_HPP_

#include <sm/boost/null_deleter.hpp>
#include "EuclideanExpression.hpp"
#include "RotationExpression.hpp"
#include "ScalarExpression.hpp"

namespace aslam {
namespace backend {

class CoordinateFrame {
 public:
  CoordinateFrame() : pp(nullptr) {};
  CoordinateFrame(const CoordinateFrame &) = default;
  CoordinateFrame(CoordinateFrame &&) = default;

  CoordinateFrame & operator = (const CoordinateFrame &) = default;
  CoordinateFrame & operator = (CoordinateFrame &&) = default;

  CoordinateFrame (const CoordinateFrame & parent, RotationExpression R_P_L = RotationExpression(), EuclideanExpression p = EuclideanExpression(), EuclideanExpression omega = EuclideanExpression(), EuclideanExpression v = EuclideanExpression(), EuclideanExpression alpha = EuclideanExpression(), EuclideanExpression a = EuclideanExpression())
    : pp(&parent, sm::null_deleter()), R_P_L(R_P_L), p(p), v(v), a(a), omega(omega), alpha(alpha)
  {
  }
  CoordinateFrame (boost::shared_ptr<const CoordinateFrame> parent, RotationExpression R_P_L = RotationExpression(), EuclideanExpression p = EuclideanExpression(), EuclideanExpression omega = EuclideanExpression(), EuclideanExpression v = EuclideanExpression(), EuclideanExpression alpha = EuclideanExpression(), EuclideanExpression a = EuclideanExpression())
    : pp(parent), R_P_L(R_P_L), p(p), v(v), a(a), omega(omega), alpha(alpha)
  {
    if(!parent){
      initGlobalsWithoutParent();
    }
  }

  CoordinateFrame (RotationExpression R_P_L, EuclideanExpression p = EuclideanExpression(), EuclideanExpression omega = EuclideanExpression(), EuclideanExpression v = EuclideanExpression(), EuclideanExpression alpha = EuclideanExpression(), EuclideanExpression a = EuclideanExpression())
    : pp(nullptr), R_P_L(R_P_L), p(p), v(v), a(a), omega(omega), alpha(alpha)
  {
    initGlobalsWithoutParent();
  }

  const boost::shared_ptr<const CoordinateFrame> getParent() const {
    return pp;
  }

  const RotationExpression & getR_G_L() const {
    if(pp && R_G_L.isEmpty()){
      R_G_L = pp->getR_G_L() * R_P_L;
    }
    return R_G_L;
  }

  const EuclideanExpression & getOmegaG() const {
    if(pp && omegaG.isEmpty()){
      omegaG = pp->getOmegaG() + pp->getR_G_L() * omega;
    }
    return omegaG;
  }

  const EuclideanExpression & getAlphaG() const {
    if(pp && alphaG.isEmpty()){
      alphaG = pp->getAlphaG() + pp->getOmegaG().cross(pp->getR_G_L() * omega) + pp->getR_G_L() * alpha;
    }
    return alphaG;
  }


  const EuclideanExpression & getPG() const {
    if(pp && pG.isEmpty()){
      pG = pp->getPG() + pp->getR_G_L() * p;
    }
    return pG;
  }

  const EuclideanExpression & getVG() const;

  const EuclideanExpression& getAG() const;

  const RotationExpression & getR_P_L() const {
    return R_P_L;
  }
  const EuclideanExpression & getPP() const {
    return p;
  }
  const EuclideanExpression & getBP() const {
    return v;
  }
  const EuclideanExpression & getAP() const {
    return a;
  }
  const EuclideanExpression & getOmegaP() const {
    return omega;
  }
  const EuclideanExpression & getAlphaP() const {
    return alpha;
  }
 private:
  void initGlobalsWithoutParent();

  boost::shared_ptr<const CoordinateFrame> pp;
  mutable RotationExpression R_G_L;
  RotationExpression R_P_L; // converting coordinates from to this to global or parent frame
  EuclideanExpression p, v, a, omega, alpha;
  mutable EuclideanExpression pG, vG, aG, omegaG, alphaG;
};


} // namespace backend
} // namespace aslam

#endif /* KINEMATICCHAIN_HPP_ */
