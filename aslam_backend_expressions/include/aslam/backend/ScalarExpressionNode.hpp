#ifndef ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP

#include <aslam/backend/JacobianContainer.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <aslam/backend/Differential.hpp>

namespace aslam {
  namespace backend {
    class ExpressionNodeVisitor;
    template <int D>
    class VectorExpressionNode;

    /**
     * \class ScalarExpressionNode
     * \brief The superclass of all classes representing scalar points.
     */
    class ScalarExpressionNode
    {
    public:
      typedef double value_t;

      ScalarExpressionNode();
      virtual ~ScalarExpressionNode();

      /// \brief Evaluate the scalar matrix.
      inline double toScalar() const;
      double evaluate() const { return toScalar(); }

      /// \brief Evaluate the Jacobians
      inline void evaluateJacobians(JacobianContainer & outJacobians) const;

      /// \brief Evaluate the Jacobians and apply the chain rule.
      template <typename DERIVED>
      EIGEN_ALWAYS_INLINE void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixBase<DERIVED> & applyChainRule) const {
        evaluateJacobians(outJacobians.apply(applyChainRule));
      }

      /// \brief Evaluate the Jacobians and apply the chain rule.
      void evaluateJacobians(JacobianContainer & outJacobians, const Differential<Eigen::Matrix<double, 1, 1>, double> & chainRuleDifferentail) const {
        evaluateJacobians(applyDifferentialToJacobianContainer(outJacobians, chainRuleDifferentail, 1));
      }

      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      virtual void accept(ExpressionNodeVisitor& visitor);  //TODO make pure and complete nodes
    protected:
      // These functions must be implemented by child classes.
      virtual double evaluateImplementation() const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;
    };


      class ScalarExpressionNodeMultiply : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeMultiply(boost::shared_ptr<ScalarExpressionNode> lhs,
                                       boost::shared_ptr<ScalarExpressionNode> rhs);
          ~ScalarExpressionNodeMultiply() override;

          void accept(ExpressionNodeVisitor& visitor) override;
      protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;
    };

      class ScalarExpressionNodeDivide : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeDivide(boost::shared_ptr<ScalarExpressionNode> lhs,
                                       boost::shared_ptr<ScalarExpressionNode> rhs);
          ~ScalarExpressionNodeDivide() override;
          void accept(ExpressionNodeVisitor& visitor) override;
      protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;

    };

    class ScalarExpressionNodeNegated : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeNegated(boost::shared_ptr<ScalarExpressionNode> rhs);
          ~ScalarExpressionNodeNegated() override;
          void accept(ExpressionNodeVisitor& visitor) override;
       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _rhs;
    };

      class ScalarExpressionNodeAdd : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeAdd(boost::shared_ptr<ScalarExpressionNode> lhs,
                                  boost::shared_ptr<ScalarExpressionNode> rhs,
                                  double multiplyRhs = 1.0);
          ~ScalarExpressionNodeAdd() override;
          void accept(ExpressionNodeVisitor& visitor) override;
       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;
          double _multiplyRhs;
    };


      class ScalarExpressionNodeConstant  : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeConstant(double s);
          ~ScalarExpressionNodeConstant() override;
          void accept(ExpressionNodeVisitor& visitor) override;
      protected:
          // These functions must be implemented by child classes.
          double evaluateImplementation() const override{return _s;}
          void evaluateJacobiansImplementation(JacobianContainer & /* outJacobians */) const override{}
          void getDesignVariablesImplementation(DesignVariable::set_t & /* designVariables */) const override{}

          double _s;
      };

      class ScalarExpressionNodeSqrt : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeSqrt(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeSqrt() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeLog : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeLog(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeLog() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeExp : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeExp(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeExp() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeAtan : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeAtan(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeAtan() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeTanh : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeTanh(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeTanh() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeAtan2 : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeAtan2(boost::shared_ptr<ScalarExpressionNode> lhs, boost::shared_ptr<ScalarExpressionNode> rhs);
          ~ScalarExpressionNodeAtan2() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          boost::shared_ptr<ScalarExpressionNode> _rhs;
      };

      class ScalarExpressionNodeAcos : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeAcos(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeAcos() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeAcosSquared : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeAcosSquared(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeAcosSquared() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
      };

      class ScalarExpressionNodeInverseSigmoid : public ScalarExpressionNode
      {
       public:
          ScalarExpressionNodeInverseSigmoid(boost::shared_ptr<ScalarExpressionNode> lhs, const double height, const double scale, const double shift);
          ScalarExpressionNodeInverseSigmoid(boost::shared_ptr<ScalarExpressionNode> lhs);
          ~ScalarExpressionNodeInverseSigmoid() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          double _height;
          double _scale;
          double _shift;
      };

      class ScalarExpressionNodePower : public ScalarExpressionNode
      {
       public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          ScalarExpressionNodePower(boost::shared_ptr<ScalarExpressionNode> lhs, const int k);
          ~ScalarExpressionNodePower() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _lhs;
          int _power;
      };


      class ScalarExpressionPiecewiseExpression : public ScalarExpressionNode
      {
       public:
          ScalarExpressionPiecewiseExpression(boost::shared_ptr<ScalarExpressionNode> e1, boost::shared_ptr<ScalarExpressionNode> e2, std::function<bool()> useFirst);
          ~ScalarExpressionPiecewiseExpression() override;

       protected:
          // These functions must be implemented by child classes.
          inline double evaluateImplementation() const override;
          inline void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<ScalarExpressionNode> _e1;
          boost::shared_ptr<ScalarExpressionNode> _e2;
          std::function<bool()> _useFirst;
      };

      template <int VectorSize, int ComponentIndex = 0>
      class ScalarExpressionNodeFromVectorExpression : public ScalarExpressionNode
      {
      public:
          ScalarExpressionNodeFromVectorExpression(boost::shared_ptr<VectorExpressionNode<VectorSize> > lhs) : _lhs(lhs){
            static_assert (ComponentIndex < VectorSize, "component index must be smaller than the vectors size");
          }
          ~ScalarExpressionNodeFromVectorExpression() override{}

       protected:
          // These functions must be implemented by child classes.
          double evaluateImplementation() const override;
          void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;
          void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

          boost::shared_ptr<VectorExpressionNode<VectorSize> > _lhs;
      };

    template <int VectorDim, int ComponentIndex>
    double ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::evaluateImplementation() const
    {
      if(!_lhs)
        return 0;
      return _lhs->evaluate()(ComponentIndex);
    }

    template <int VectorDim, int ComponentIndex>
    void ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
    {
      if(_lhs)
        _lhs->evaluateJacobians(outJacobians, Eigen::Matrix<double, 1, VectorDim>::Unit(ComponentIndex));
    }

    template <int VectorDim, int ComponentIndex>
    void ScalarExpressionNodeFromVectorExpression<VectorDim, ComponentIndex>::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
    {
      if(_lhs)
        _lhs->getDesignVariables(designVariables);
    }

  } // namespace backend
} // namespace aslam


#include "implementation/ScalarExpressionNode.hpp"

#endif /* ASLAM_BACKEND_EUCLIDEAN_EXPRESSION_NODE_HPP */
