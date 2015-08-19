#ifndef ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP

#include <aslam/backend/JacobianContainer.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <aslam/backend/VectorExpressionNode.hpp>

namespace aslam {
  namespace backend {


    /**
     * \class ScalarExpressionNode
     * \brief The superclass of all classes representing scalar points.
     */
    class ScalarExpressionNode
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      ScalarExpressionNode();
      virtual ~ScalarExpressionNode();

      /// \brief Evaluate the scalar matrix.
      double toScalar() const;
      double evaluate() const { return toScalar(); }

      /// \brief Evaluate the Jacobians
      void evaluateJacobians(JacobianContainer & outJacobians) const;

      /// \brief Evaluate the Jacobians and apply the chain rule.
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;

      /// \brief Evaluate the Jacobians and apply the chain rule.
      void evaluateJacobians(JacobianContainer & outJacobians, const Differential<Eigen::Matrix<double, 1, 1>, double> & applyChainRule) const {
        Eigen::VectorXd m;
        applyChainRule.applyInto(Eigen::Matrix<double, 1,1>::Ones(), m);
        evaluateJacobians(outJacobians, m);
      }

      void getDesignVariables(DesignVariable::set_t & designVariables) const;
    protected:
      // These functions must be implemented by child classes.
      virtual double toScalarImplementation() const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const = 0;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;
    };


      class ScalarExpressionNodeMultiply : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeMultiply(boost::shared_ptr<ScalarExpressionNode> lhs,
                                       boost::shared_ptr<ScalarExpressionNode> rhs);
          virtual ~ScalarExpressionNodeMultiply();
      protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;

    };

      class ScalarExpressionNodeDivide : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeDivide(boost::shared_ptr<ScalarExpressionNode> lhs,
                                       boost::shared_ptr<ScalarExpressionNode> rhs);
          virtual ~ScalarExpressionNodeDivide();
      protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;

    };

    class ScalarExpressionNodeNegated : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeNegated(boost::shared_ptr<ScalarExpressionNode> rhs);
          virtual ~ScalarExpressionNodeNegated();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

          boost::shared_ptr<ScalarExpressionNode> _rhs;
    };

      class ScalarExpressionNodeAdd : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeAdd(boost::shared_ptr<ScalarExpressionNode> lhs,
                                  boost::shared_ptr<ScalarExpressionNode> rhs,
                                  double multiplyRhs = 1.0);
          virtual ~ScalarExpressionNodeAdd();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;
          double _multiplyRhs;
    };


      class ScalarExpressionNodeConstant  : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeConstant(double s);
          virtual ~ScalarExpressionNodeConstant();

      protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const{return _s;}
          virtual void evaluateJacobiansImplementation(JacobianContainer & /* outJacobians */) const{}
          virtual void evaluateJacobiansImplementation(JacobianContainer & /* outJacobians */, const Eigen::MatrixXd & /* applyChainRule */) const{}
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & /* designVariables */) const{}

          double _s;
      };

      class ScalarExpressionNodeSqrt : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeSqrt(boost::shared_ptr<ScalarExpressionNode> lhs);
          virtual ~ScalarExpressionNodeSqrt();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeLog : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeLog(boost::shared_ptr<ScalarExpressionNode> lhs);
          virtual ~ScalarExpressionNodeLog();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeExp : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeExp(boost::shared_ptr<ScalarExpressionNode> lhs);
          virtual ~ScalarExpressionNodeExp();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeAtan : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeAtan(boost::shared_ptr<ScalarExpressionNode> lhs);
          virtual ~ScalarExpressionNodeAtan();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeAcos : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeAcos(boost::shared_ptr<ScalarExpressionNode> lhs);
          virtual ~ScalarExpressionNodeAcos();

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const override;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      template <int VectorSize, int ComponentIndex = 0>
      class ScalarExpressionNodeFromVectorExpression : public ScalarExpressionNode
      {
      public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodeFromVectorExpression(boost::shared_ptr<VectorExpressionNode<VectorSize> > lhs) : _lhs(lhs){
            static_assert (ComponentIndex < VectorSize, "component index must be smaller than the vectors size");
          }
          virtual ~ScalarExpressionNodeFromVectorExpression(){}

       protected:
          // These functions must be implemented by child classes.
          virtual double toScalarImplementation() const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
          virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
          virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

          boost::shared_ptr<VectorExpressionNode<VectorSize> > _lhs;
      };

    template <int VectorDim, int ComponentIndex>
    double ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::toScalarImplementation() const
    {
        return _lhs->evaluate()(ComponentIndex);
    }

    template <int VectorDim, int ComponentIndex>
    void ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
        _lhs->evaluateJacobians(outJacobians, Eigen::Matrix<double, 1, VectorDim>::Unit(ComponentIndex));
    }

    template <int VectorDim, int ComponentIndex>
    void ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
    {
        _lhs->evaluateJacobians(outJacobians, applyChainRule * Eigen::Matrix<double, 1, VectorDim>::Unit(ComponentIndex));
    }

    template <int VectorDim, int ComponentIndex>
    void ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
        _lhs->getDesignVariables(designVariables);
    }

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_EUCLIDEAN_EXPRESSION_NODE_HPP */
