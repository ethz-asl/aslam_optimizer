#ifndef ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP

#include <type_traits>

#include <aslam/backend/JacobianContainer.hpp>
#include <aslam/backend/Differential.hpp>
#include <aslam/backend/ExpressionNodeVisitor.hpp>
#include <aslam/backend/ScalarExpression.hpp>

namespace aslam {
  namespace backend {
    
    template<int D>
    class VectorExpressionNode
    {
    public:
      typedef Eigen::Matrix<double,D,1> vector_t;
      typedef vector_t value_t;
      typedef Differential<vector_t, double> differential_t;

      VectorExpressionNode() = default;
      virtual ~VectorExpressionNode() = default;
      
      vector_t evaluate() const { return evaluateImplementation(); }
      vector_t toVector() const { return evaluate(); }
      
      void evaluateJacobians(JacobianContainer & outJacobians) const;
      template <typename DERIVED>
      EIGEN_ALWAYS_INLINE void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixBase<DERIVED> & applyChainRule) const {
        evaluateJacobians(outJacobians.apply(applyChainRule));
      }

      void evaluateJacobians(JacobianContainer & outJacobians, const differential_t & chainRuleDifferentail) const;
      void getDesignVariables(DesignVariable::set_t & designVariables) const;

      virtual int getSize() const { assert(D != Eigen::Dynamic); return D; }

      virtual void accept(ExpressionNodeVisitor& visitor) { visitor.visit("V", this); }; //TODO make pure and complete nodes
    private:
      virtual vector_t evaluateImplementation() const = 0;
      virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const = 0;
      virtual void evaluateJacobiansImplementationWithDifferential(JacobianContainer & outJacobians, const differential_t & chainRuleDifferentail) const;
      virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const = 0;
    };

    template <int D>
    class ConstantVectorExpressionNode : public VectorExpressionNode<D> {
     public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef typename VectorExpressionNode<D>::vector_t vector_t;
      typedef typename VectorExpressionNode<D>::differential_t differential_t;

      ConstantVectorExpressionNode(int rows = D, int cols = 1) {
        static_cast<void>(rows); static_cast<void>(cols); // necessary to prevent warnings for release build;
        if (D != Eigen::Dynamic){
          SM_ASSERT_EQ_DBG(std::runtime_error, rows, D, "dynamic size has to equal static size");
        }
        SM_ASSERT_EQ_DBG(std::runtime_error, cols, 1, "there is only one column supported as vector expression.");
      }
      ConstantVectorExpressionNode(const vector_t & value) : value(value) {}

      ~ConstantVectorExpressionNode() override = default;
      int getSize() const override { return value.rows(); }

      void accept(ExpressionNodeVisitor& visitor) override;
     private:
      vector_t evaluateImplementation() const override { return value; }
      void evaluateJacobiansImplementation(JacobianContainer &) const override {}
      void getDesignVariablesImplementation(DesignVariable::set_t &) const override {}
     private:
      vector_t value;
    };

    template <int D>
    class StackedScalarVectorExpressionNode : public VectorExpressionNode<D> {
     public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef typename VectorExpressionNode<D>::vector_t vector_t;
      typedef typename VectorExpressionNode<D>::differential_t differential_t;

      template <typename... Args>
      StackedScalarVectorExpressionNode(Args&&... args) : components{args...} {
        static_assert(sizeof...(args) == D);
        static_assert(std::is_convertible<typename std::common_type<Args...>::type, boost::shared_ptr<ScalarExpressionNode>>::value);
      }

      ~StackedScalarVectorExpressionNode() override = default;
      int getSize() const override {
        return components.size();
      }

      void accept(ExpressionNodeVisitor& visitor) override;

     private:
      vector_t evaluateImplementation() const override;
      void evaluateJacobiansImplementation(JacobianContainer&) const override;

      void getDesignVariablesImplementation(
          DesignVariable::set_t&) const override;

     private:
      std::array<boost::shared_ptr<ScalarExpressionNode>, D> components;
    };

    template <int D>
    class VectorExpressionNodeAddVector : public VectorExpressionNode<D> {
     public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef typename VectorExpressionNode<D>::vector_t vector_t;
      typedef typename VectorExpressionNode<D>::differential_t differential_t;

      VectorExpressionNodeAddVector(
          boost::shared_ptr<VectorExpressionNode<D>> lhs,
          boost::shared_ptr<VectorExpressionNode<D>> rhs)
          : _lhs(lhs), _rhs(rhs) {}
      ~VectorExpressionNodeAddVector() override = default;

      void accept(ExpressionNodeVisitor& visitor) override;

     private:
      vector_t evaluateImplementation() const override;
      void evaluateJacobiansImplementation(
          JacobianContainer& outJacobians) const override;
      void getDesignVariablesImplementation(
          DesignVariable::set_t& designVariables) const override;

      boost::shared_ptr<VectorExpressionNode<D>> _lhs;
      boost::shared_ptr<VectorExpressionNode<D>> _rhs;
    };

    template <int D>
    class VectorExpressionNodeScalarMultiply
        : public VectorExpressionNode<D> {
     public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef typename VectorExpressionNode<D>::vector_t vector_t;
      typedef typename VectorExpressionNode<D>::differential_t differential_t;

      VectorExpressionNodeScalarMultiply(
          boost::shared_ptr<VectorExpressionNode<D>> p,
          boost::shared_ptr<ScalarExpressionNode> s): _p(p), _s(s) {}
      ~VectorExpressionNodeScalarMultiply() override = default;

     private:
      vector_t evaluateImplementation() const override;
      void evaluateJacobiansImplementation(
          JacobianContainer& outJacobians) const override;
      void getDesignVariablesImplementation(
          DesignVariable::set_t& designVariables) const override;

      boost::shared_ptr<VectorExpressionNode<D>> _p;
      boost::shared_ptr<ScalarExpressionNode> _s;
    };

  } // namespace backend
} // namespace aslam

#include "implementation/VectorExpressionNode.hpp"

#endif /* ASLAM_BACKEND_VECTOR_EXPRESSION_NODE_HPP */
