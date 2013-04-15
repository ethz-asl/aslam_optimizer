#include <aslam/backend/EuclideanPoint.hpp>


namespace aslam {
  namespace backend {
    EuclideanPoint::EuclideanPoint(const Eigen::Vector3d & p) :
      _p(p), _p_p(p)
    {

    }
    EuclideanPoint::~EuclideanPoint()
    {

    }

    /// \brief Revert the last state update.
    void EuclideanPoint::revertUpdateImplementation()
    {
      _p = _p_p;
    }
    
    /// \brief Update the design variable.
    void EuclideanPoint::updateImplementation(const double * dp, int size)
    {
        SM_ASSERT_EQ_DBG(std::runtime_error, size, 3, "Incorrect size");
      _p_p = _p;
      
      Eigen::Map< const Eigen::Vector3d > dpv(dp);
      _p += dpv;

    }
    
    /// \brief the size of an update step
    int EuclideanPoint::minimalDimensionsImplementation() const
    {
      return 3;
    }
    
    Eigen::Vector3d EuclideanPoint::toEuclideanImplementation()
    {
      return _p;
    }
    
    void EuclideanPoint::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      outJacobians.add(const_cast<EuclideanPoint *>(this), Eigen::Matrix3d::Identity());
    }
    
    void EuclideanPoint::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
      outJacobians.add(const_cast<EuclideanPoint *>(this), applyChainRule);
    }

    void EuclideanPoint::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      designVariables.insert(const_cast<EuclideanPoint *>(this));
    }

    EuclideanExpression EuclideanPoint::toExpression()
    {
      return EuclideanExpression(this);
    }

    void EuclideanPoint::getParametersImplementation(
        Eigen::MatrixXd& value) const {
      value = _p;
    }

    void EuclideanPoint::setParametersImplementation(
        const Eigen::MatrixXd& value) {
      _p_p = _p;
      _p = value;
    }

  } // namespace backend
} // namespace aslam
