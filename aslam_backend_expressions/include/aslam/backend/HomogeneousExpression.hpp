#ifndef ASLAM_HOMOGENEOUS_EXPRESSION_HPP
#define ASLAM_HOMOGENEOUS_EXPRESSION_HPP

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include "EuclideanExpression.hpp"
#include <aslam/backend/JacobianContainer.hpp>
#include <set>

namespace aslam {
  namespace backend {
    class HomogeneousExpressionNode;
    class EuclideanExpression;

    class HomogeneousExpression
    {
    public:
        SM_DEFINE_EXCEPTION(Exception, std::runtime_error);

      HomogeneousExpression();
      HomogeneousExpression(HomogeneousExpressionNode * designVariable);
      HomogeneousExpression(boost::shared_ptr<HomogeneousExpressionNode> designVariable);
        // Create a constant expression
        HomogeneousExpression(const Eigen::Vector4d & p);
        HomogeneousExpression(const Eigen::Vector3d & p);

      virtual ~HomogeneousExpression();
      
      Eigen::Vector4d toHomogeneous();
      Eigen::Vector4d toValue() { return toHomogeneous(); }
      EuclideanExpression toEuclideanExpression();
      void evaluateJacobians(JacobianContainer & outJacobians) const;
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      void getDesignVariables(DesignVariable::set_t & designVariables) const;
      boost::shared_ptr<HomogeneousExpressionNode> root() { return _root; }

    private:
      friend class TransformationExpression;
      boost::shared_ptr<HomogeneousExpressionNode> _root;
    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_HOMOGENEOUS_EXPRESSION_HPP */
