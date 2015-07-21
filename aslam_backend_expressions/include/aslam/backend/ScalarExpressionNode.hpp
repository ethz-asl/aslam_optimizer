#ifndef ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP
#define ASLAM_BACKEND_SCALAR_EXPRESSION_NODE_HPP

#include <aslam/backend/JacobianContainer.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <aslam/backend/VectorExpressionNode.hpp>
#include <aslam/backend/MatrixExpressionNode.hpp>
#include <aslam/backend/GenericMatrixExpressionNode.hpp>

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

      /// \brief Evaluate the Jacobians
      void evaluateJacobians(JacobianContainer & outJacobians) const;

      /// \brief Evaluate the Jacobians and apply the chain rule.
      void evaluateJacobians(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
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


    template <int Rows, int Cols, int RowIndex = 0, int ColIndex = 0>
    class ScalarExpressionNodeFromMatrixExpression : public ScalarExpressionNode
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ScalarExpressionNodeFromMatrixExpression(boost::shared_ptr< GenericMatrixExpressionNode<Rows,Cols,double> > lhs) : _lhs(lhs){
          static_assert (RowIndex < Rows, "Row index index must be smaller than the number of rows");
          static_assert (ColIndex < Cols, "Column index index must be smaller than the number of columns");
        }
        virtual ~ScalarExpressionNodeFromMatrixExpression(){}

     protected:
        // These functions must be implemented by child classes.
        virtual double toScalarImplementation() const;
        virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const;
        virtual void evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const;
        virtual void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const;

        boost::shared_ptr< GenericMatrixExpressionNode<Rows,Cols,double> > _lhs;
  };

  template <int Rows, int Cols, int RowIndex, int ColIndex>
  double ScalarExpressionNodeFromMatrixExpression<Rows, Cols, RowIndex, ColIndex>::toScalarImplementation() const
  {
      return _lhs->evaluate()(RowIndex, ColIndex);
  }

  template <int Rows, int Cols, int RowIndex, int ColIndex>
  void ScalarExpressionNodeFromMatrixExpression<Rows, Cols, RowIndex, ColIndex>::evaluateJacobiansImplementation(JacobianContainer & outJacobians) const
  {

      typedef GenericMatrixExpressionNode<Rows,Cols,double> GME;

      Eigen::Matrix<double, Rows, Cols> J = Eigen::Matrix<double, Rows, Cols>::Zero();
      J(RowIndex, ColIndex) = 1.;
      MatrixDifferential<typename GME::matrix_t::Scalar, typename GME::matrix_t> D(J);

      _lhs->evaluateJacobians(outJacobians, D);
  }

  template <int Rows, int Cols, int RowIndex, int ColIndex>
  void ScalarExpressionNodeFromMatrixExpression<Rows, Cols, RowIndex, ColIndex>::evaluateJacobiansImplementation(JacobianContainer & outJacobians, const Eigen::MatrixXd & applyChainRule) const
  {
    typedef GenericMatrixExpressionNode<Rows,Cols,double> GME;

    Eigen::Matrix<double, Rows, Cols> J = Eigen::Matrix<double, Rows, Cols>::Zero();
    J(RowIndex, ColIndex) = 1.;
    J = applyChainRule*J;
    MatrixDifferential<typename GME::matrix_t::Scalar, typename GME::matrix_t> D(J);

    _lhs->evaluateJacobians(outJacobians, D);
  }

  template <int Rows, int Cols, int RowIndex, int ColIndex>
  void ScalarExpressionNodeFromMatrixExpression<Rows, Cols, RowIndex, ColIndex>::getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const
  {
      _lhs->getDesignVariables(designVariables);
  }

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_EUCLIDEAN_EXPRESSION_NODE_HPP */
