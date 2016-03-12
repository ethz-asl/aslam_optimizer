#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/JacobianContainerSparse.hpp>
#include <aslam/backend/JacobianContainerDense.hpp>
#include <boost/shared_ptr.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
using namespace boost::python;
using namespace aslam::backend;


DesignVariable * (JacobianContainerSparse::*jc_dvptr)(size_t ) = &JacobianContainerSparse::designVariable;

SparseBlockMatrix (JacobianContainerSparse::*jc_asm1)() const = &JacobianContainerSparse::asSparseMatrix;

SparseBlockMatrix jc_asm2(JacobianContainerSparse * jc, const Eigen::VectorXi & colBlockIndices)
{
  std::vector<int> cbi;
  cbi.insert(cbi.begin(), &colBlockIndices[0], &colBlockIndices[0] + colBlockIndices.size());
  return jc->asSparseMatrix(cbi);
}

Eigen::MatrixXd (JacobianContainerSparse::*jc_adm1)() const = &JacobianContainerSparse::asDenseMatrix;

Eigen::MatrixXd jc_adm2(JacobianContainerSparse * jc, const Eigen::VectorXi & colBlockIndices)
{
  std::vector<int> cbi;
  cbi.insert(cbi.begin(), &colBlockIndices[0], &colBlockIndices[0] + colBlockIndices.size());
  return jc->asDenseMatrix(cbi);
}

void addWrapper(JacobianContainer& jc, DesignVariable* designVariable, const Eigen::MatrixXd& mat) {
  jc.add(designVariable, mat);
}

void addContainerWrapper(JacobianContainerSparse& jc, const JacobianContainerSparse& rhs, const Eigen::MatrixXd& applyChainRule = Eigen::MatrixXd(0,0)) {
  if (applyChainRule.size() == 0)
    jc.add<Eigen::MatrixXd>(rhs, nullptr);
  else
    jc.add(rhs, &applyChainRule);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(JacobianContainerSparse_add_overloads, addContainerWrapper, 2, 3);

void exportJacobianContainer()
{

  class_<JacobianContainer, boost::shared_ptr<JacobianContainer>, boost::noncopyable>("JacobianContainer", no_init)
    .def("add", pure_virtual(&addWrapper))
    .def("applyChainRule", pure_virtual(&JacobianContainer::applyChainRule))
    .def("asDenseMatrix", pure_virtual(&JacobianContainer::asDenseMatrix))
    .def("clear", pure_virtual(&JacobianContainer::clear))
    .def("isFinite", pure_virtual(&JacobianContainer::isFinite))
    .def("rows", &JacobianContainer::rows)
    .def("setScale", &JacobianContainer::setScale)
  ;

  typedef JacobianContainerDense<Eigen::MatrixXd> JCDense;
  class_<JCDense, bases<JacobianContainer> >("JacobianContainerDense", init<int,int>("JacobianContainerDense(int rows, int cols): Constructor"))
    .def("Jacobian", &JCDense::Jacobian)
  ;

  class_<JacobianContainerSparse, bases<JacobianContainer> >("JacobianContainerSparse", init<int>())
      
    //void evaluateHessian(const Eigen::VectorXd & e, const Eigen::MatrixXd & invR, SparseBlockMatrix & outHessian, Eigen::VectorXd & outRhs) const;
      
    .def("add", &addWrapper) // hack: otherwise the base method is not available any more
    .def("add", &addContainerWrapper, JacobianContainerSparse_add_overloads())

    /// \brief how many design variables does this jacobian container represent.
    .def("numDesignVariables", &JacobianContainerSparse::numDesignVariables)
            
    /// \brief Get design variable i.
    .def("designVariable", jc_dvptr, return_internal_reference<>())
      
    /// Get the Jacobian associated with a particular design variable.
    .def("Jacobian", &JacobianContainerSparse::Jacobian, return_value_policy<copy_const_reference>())
      
    /// \brief How many rows does this set of Jacobians have?
    .def("rows", &JacobianContainerSparse::rows)

      /// \brief Clear the contents of this container
    .def("reset", &JacobianContainerSparse::reset)

      
    /// \brief Gets a sparse matrix with the Jacobians. The matrix is, in fact, dense
    ///        and the Jacobian ordering matches the sort order. 
    /// 
    ///        \todo add a function that allows us to get the full block row
    ///        Jacobian. This is tricky because you have to know the
    ///        full block structure of the Hessian
    .def("asSparseMatrix", jc_asm1)
    /// \brief Get a sparse matrix with the Jacobians. This uses the column block indices
    ///        to build the sparse, full width jacobian matrix
    .def("asSparseMatrix", jc_asm2)

    .def("asDenseMatrix", jc_adm1)
    .def("asDenseMatrix", jc_adm2)
    
    /// The number of columns in the compressed Jacobian. Warning: this is expensive.
    .def("cols", &JacobianContainerSparse::cols)
    ;
    }
