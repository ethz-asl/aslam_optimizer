#ifndef ASLAM_BACKEND_EUCLIDEAN_POINT_HPP
#define ASLAM_BACKEND_EUCLIDEAN_POINT_HPP

#include "EuclideanExpressionNode.hpp"
#include "EuclideanExpression.hpp"
#include <aslam/backend/DesignVariable.hpp>


namespace aslam {
  namespace backend {
    
    class EuclideanPoint : public EuclideanExpressionNode, public DesignVariable
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      EuclideanPoint(const Eigen::Vector3d & p);
      virtual ~EuclideanPoint();

      /// \brief Revert the last state update.
      virtual void revertUpdateImplementation();

      /// \brief Update the design variable.
      virtual void updateImplementation(const double * dp, int size);

      /// \brief the size of an update step
      virtual int minimalDimensionsImplementation() const;

      EuclideanExpression toExpression();
    private:
      virtual Eigen::Vector3d toEuclideanImplementation();

      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;

      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

      /// \brief The current value of the design variable.
      Eigen::Vector3d _p;

      /// \brief The previous version of the design variable.
      Eigen::Vector3d _p_p;


    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_EUCLIDEAN_POINT_HPP */