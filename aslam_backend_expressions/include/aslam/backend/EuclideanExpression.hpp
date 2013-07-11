#ifndef ASLAM_EUCLIDEAN_EXPRESSION_HPP
#define ASLAM_EUCLIDEAN_EXPRESSION_HPP

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <aslam/backend/JacobianContainer.hpp>
#include "HomogeneousExpression.hpp"
#include "VectorExpression.hpp"
#include <set>


namespace aslam {
  namespace backend {
    class HomogeneousExpression;
    class EuclideanExpressionNode;
    
    class EuclideanExpression
    {
    public:
        SM_DEFINE_EXCEPTION(Exception, std::runtime_error);

      EuclideanExpression(EuclideanExpressionNode * designVariable);
      EuclideanExpression(boost::shared_ptr<EuclideanExpressionNode> designVariable);
      EuclideanExpression(const VectorExpression<3> & vectorExpression);
      virtual ~EuclideanExpression();
      
      Eigen::Vector3d toEuclidean();
      Eigen::Vector3d toValue() { return toEuclidean(); }
      HomogeneousExpression toHomogeneousExpression();

      void evaluateJacobians(JacobianContainer & outJacobians) const;
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      EuclideanExpression cross(const EuclideanExpression & p);
      EuclideanExpression operator+(const EuclideanExpression & p);
      EuclideanExpression operator-(const EuclideanExpression & p);
      EuclideanExpression operator-(const Eigen::Vector3d & p);

      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      boost::shared_ptr<EuclideanExpressionNode> root() { return _root; }
    private:
      /// \todo make the default constructor private.
      EuclideanExpression();

      friend class RotationExpression;
      friend class MatrixExpression;
      
      boost::shared_ptr<EuclideanExpressionNode> _root;
    };
    
  } // namespace backend
} // namespace aslam


#endif /* ASLAM_EUCLIDEAN_EXPRESSION_HPP */
