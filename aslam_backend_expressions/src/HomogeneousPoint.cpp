#include <aslam/backend/HomogeneousPoint.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>


namespace aslam {
  namespace backend {
    HomogeneousPoint::HomogeneousPoint(const Eigen::Vector4d & p) :
      _p(p), _p_p(p)
    {
      double recipPnorm = 1.0/p.norm();
      _p *= recipPnorm;
      _p_p *= recipPnorm;
    }
    HomogeneousPoint::~HomogeneousPoint()
    {

    }

    /// \brief Revert the last state update.
    void HomogeneousPoint::revertUpdateImplementation()
    {
      _p = _p_p;
    }
    
    /// \brief Update the design variable.
    void HomogeneousPoint::updateImplementation(const double * dp, int size)
    {
      _p_p = _p;
      Eigen::Map<const Eigen::Vector3d> dpv(dp);
      _p = sm::kinematics::updateQuat(_p, dpv);      

    }
    
    /// \brief the size of an update step
    int HomogeneousPoint::minimalDimensionsImplementation() const
    {
      return 3;
    }
    
    Eigen::Vector4d HomogeneousPoint::toHomogeneousImplementation()
    {
      return _p;
    }
    
    void HomogeneousPoint::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      outJacobians.add(const_cast<HomogeneousPoint *>(this), sm::kinematics::quatInvS(_p));
    }
    
    void HomogeneousPoint::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      outJacobians.add(const_cast<HomogeneousPoint *>(this), applyChainRule * sm::kinematics::quatInvS(_p));
    }
      
    void HomogeneousPoint::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      designVariables.insert(const_cast<HomogeneousPoint *>(this));
    }

    HomogeneousExpression HomogeneousPoint::toExpression()
    {
      return HomogeneousExpression(this);
    }

  } // namespace backend
} // namespace aslam
