
#include <aslam/backend/MappedRotationQuaternion.hpp>
#include <sm/kinematics/rotations.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>


namespace aslam {
  namespace backend {

    MappedRotationQuaternion::MappedRotationQuaternion(double * q) : _q(q), _p_q(_q), _C(sm::kinematics::quat2r(_q)) {}

    MappedRotationQuaternion::~MappedRotationQuaternion(){}

      
    /// \brief Revert the last state update.
    void MappedRotationQuaternion::revertUpdateImplementation()
    {
      _q = _p_q;
      _C = sm::kinematics::quat2r(_q);
    }
    
    /// \brief Update the design variable.
    void MappedRotationQuaternion::updateImplementation(const double * dp, int size) 
    {
      SM_ASSERT_EQ_DBG(Exception, size, 3, "Incorrect update size");
      _p_q = _q;
      Eigen::Map<const Eigen::Vector3d> dpv(dp);
      _q = sm::kinematics::updateQuat(_q, dpv);
      _C = sm::kinematics::quat2r(_q);
    }
    
    int MappedRotationQuaternion::minimalDimensionsImplementation() const
    { 
      return 3; 
    }
    
    Eigen::Matrix3d MappedRotationQuaternion::toRotationMatrixImplementation()
    {
      return _C;
    }

    void MappedRotationQuaternion::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      outJacobians.add( const_cast<MappedRotationQuaternion *>(this), Eigen::Matrix3d::Identity() );
    }

    void MappedRotationQuaternion::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      outJacobians.add( const_cast<MappedRotationQuaternion*>(this), applyChainRule );
    }

    RotationExpression MappedRotationQuaternion::toExpression()
    {
      return RotationExpression(this);
    }

    void MappedRotationQuaternion::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      designVariables.insert(const_cast<MappedRotationQuaternion*>(this));
    }

    void MappedRotationQuaternion::getParametersImplementation(
        Eigen::MatrixXd& value) const {
      value = _q;
    }

    void MappedRotationQuaternion::setParametersImplementation(
        const Eigen::MatrixXd& value) {
      _p_q = _q;
      _q = value;
      _C = sm::kinematics::quat2r(_q);
    }

  } // namespace backend
} // namespace aslam

