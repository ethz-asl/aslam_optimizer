#include <aslam/backend/EuclideanExpression.hpp>
#include <aslam/backend/EuclideanExpressionNode.hpp>
#include <sm/boost/null_deleter.hpp>

namespace aslam {
  namespace backend {
    EuclideanExpression::EuclideanExpression()
    {
      
    }

    EuclideanExpression::EuclideanExpression(EuclideanExpressionNode * designVariable) :
      _root( designVariable, sm::null_deleter() )
    {
      
    }

    EuclideanExpression::EuclideanExpression(boost::shared_ptr<EuclideanExpressionNode> root) :
      _root(root)
    {

    }

    EuclideanExpression::EuclideanExpression(const VectorExpression<3> & vectorExpression) :
      _root(new VectorExpression2EuclideanExpressionAdapter(vectorExpression.root()))
    {

    }

    EuclideanExpression::~EuclideanExpression()
    {
      
    }
  
      
    Eigen::Vector3d EuclideanExpression::toEuclidean()
    {
      return _root->toEuclidean();
    }

    HomogeneousExpression EuclideanExpression::toHomogeneousExpression()
    {
      // \todo
        SM_THROW(Exception, "Not Implemented");
      return HomogeneousExpression();
    }
    
    void EuclideanExpression::evaluateJacobians(JacobianContainer & outJacobians) const
    {
      _root->evaluateJacobians(outJacobians);
    }

    void EuclideanExpression::evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
        _root->evaluateJacobians(outJacobians, applyChainRule);
    }

    EuclideanExpression EuclideanExpression::cross(const EuclideanExpression & p)
    {
      boost::shared_ptr<EuclideanExpressionNode> newRoot( new EuclideanExpressionNodeCrossEuclidean(_root, p._root));
      return EuclideanExpression(newRoot);
    }

    EuclideanExpression EuclideanExpression::operator+(const EuclideanExpression & p)
    {
      boost::shared_ptr<EuclideanExpressionNode> newRoot( new EuclideanExpressionNodeAddEuclidean(_root, p._root));
      return EuclideanExpression(newRoot);
    }

    EuclideanExpression EuclideanExpression::operator-(const EuclideanExpression & p)
    {
      boost::shared_ptr<EuclideanExpressionNode> newRoot( new EuclideanExpressionNodeSubtractEuclidean(_root, p._root));
      return EuclideanExpression(newRoot);
    }

    EuclideanExpression EuclideanExpression::operator-(const Eigen::Vector3d & p)
    {
      boost::shared_ptr<EuclideanExpressionNode> newRoot( new EuclideanExpressionNodeSubtractVector(_root, p));
      return EuclideanExpression(newRoot);
    }

    void EuclideanExpression::getDesignVariables(DesignVariable::set_t & designVariables) const
    {
      _root->getDesignVariables(designVariables);
    }

  
  } // namespace backend
} // namespace aslam
