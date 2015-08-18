#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <boost/shared_ptr.hpp>
#include <sm/PropertyTree.hpp>

// some wrappers:
Eigen::VectorXd b(const aslam::backend::Optimizer * o)
{
	return o->b();
}
Eigen::VectorXd dx(const aslam::backend::Optimizer * o)
{
	return o->dx();
}
Eigen::VectorXd rhs(const aslam::backend::Optimizer * o)
{
	return o->rhs();
}

void exportOptimizer()
{
    using namespace boost::python;
    using namespace aslam::backend;



 
    class_<SolutionReturnValue>("SolutionReturnValue")
        .def_readwrite("JStart",&SolutionReturnValue::JStart)
        .def_readwrite("JFinal",&SolutionReturnValue::JFinal)
        .def_readwrite("iterations",&SolutionReturnValue::iterations)
        .def_readwrite("failedIterations",&SolutionReturnValue::failedIterations)
        .def_readwrite("lmLambdaFinal",&SolutionReturnValue::lmLambdaFinal)
        .def_readwrite("dXFinal",&SolutionReturnValue::dXFinal)
        .def_readwrite("dJFinal",&SolutionReturnValue::dJFinal)
        .def_readwrite("linearSolverFailure",&SolutionReturnValue::linearSolverFailure)
        ;





    class_<Optimizer, boost::shared_ptr<Optimizer> >("Optimizer",init<>())
        .def(init<OptimizerOptions>())
        .def("setProblem", static_cast<void(Optimizer::*)(boost::shared_ptr<OptimizationProblemBase>)>(&Optimizer::setProblem))

        /// \brief initialize the optimizer to run on an optimization problem.
        ///        This should be called before calling optimize()
        .def("initialize", &Optimizer::initialize)
      
        /// \brief initialize the linear solver specified in the optimizer options.
        .def("initializeLinearSolver", &Optimizer::initializeLinearSolver)

        /// \brief Run the optimization
        .def("optimize", &Optimizer::optimize)
        .def("optimizeDogLeg", &Optimizer::optimizeDogLeg)

        .def("buildGnMatrices", &Optimizer::buildGnMatrices)
        /// \brief Get the optimizer options.
        .add_property("options", make_function(&Optimizer::options,return_internal_reference<>()))

        /// \brief return the full Hessian
        .def("H", &Optimizer::H, return_internal_reference<>())
      
        /// \brief return the full rhs
        .def("rhs", &rhs)

        /// \brief return the reduced system lhs
        .def("A", &Optimizer::A, return_internal_reference<>())
      
        /// \brief return the reduced system rhs
        .def("b", &b)

        /// \brief return the reduced system dx
        .def("dx", &dx)


        /// The value of the objective function.
        .def("J", &Optimizer::J)

        // \todo Covariance calculations
        // void computeCovariances();
        // const Eigen::MatrixXd & getDenseBlockCovariance(int di1, int di2);
        // Eigen::MatrixXd getSparseSparseCovariance(int si1, int si2);
        // Eigen::MatrixXd getDenseSparseCovariance(int di, int si);

        /// \brief Evaluate the error at the current state.
        .def("evaluateError", &Optimizer::evaluateError)

        /// \brief Get dense design variable i.
        .def("denseVariable", &Optimizer::denseVariable, return_internal_reference<>())
        /// \brief Get sparse design variable i
        .def("sparseVariable", &Optimizer::sparseVariable, return_internal_reference<>())

        /// \brief how many dense design variables are involved in the problem
        .def("numDenseDesignVariables", &Optimizer::numDenseDesignVariables)

        /// \brief how many sparse design variables are involved in the problem
        .def("numSparseDesignVariables", &Optimizer::numSparseDesignVariables)

        .def("printTiming", &Optimizer::printTiming)

        .def("computeCovariances", &Optimizer::computeCovariances)
        .def("computeDiagonalCovariances", &Optimizer::computeDiagonalCovariances)
        // \todo Think of a nice way to expose this to Python
        //.def("computeCovarianceBlocks", &Optimizer::computeCovarianceBlocks)
        .def("getCovarianceBlock", &Optimizer::getCovarianceBlock,return_internal_reference<>())
        .def("P", &Optimizer::P,return_internal_reference<>())
        .def("getCovariance", &Optimizer::getCovariance,return_internal_reference<>())
            
        ;


    class_<Optimizer2, boost::shared_ptr<Optimizer2> >("Optimizer2",init<>())
        .def(init<Optimizer2Options>())
        .def("setProblem", &Optimizer2::setProblem)

        /// \brief initialize the optimizer to run on an optimization problem.
        ///        This should be called before calling optimize()
        .def("initialize", &Optimizer2::initialize)
      
        /// \brief initialize the linear solver specified in the optimizer options.
        .def("initializeLinearSolver", &Optimizer2::initializeLinearSolver)

        /// \brief Run the optimization
        .def("optimize", &Optimizer2::optimize)
        //.def("optimizeDogLeg", &Optimizer2::optimizeDogLeg)

        /// \brief Get the optimizer options.
        .add_property("options", make_function(&Optimizer2::options,return_internal_reference<>()))

        /// \brief return the full Hessian
        //.def("H", &Optimizer2::H, return_internal_reference<>())
      
        /// \brief return the full rhs
        //.def("rhs", &rhs)

        /// \brief return the reduced system lhs
        //.def("A", &Optimizer2::A, return_internal_reference<>())
      
        /// \brief return the reduced system rhs
        //.def("b", &b)

        /// \brief return the reduced system dx
        //.def("dx", &dx)


        /// The value of the objective function.
        .def("J", &Optimizer2::J)

        // \todo Covariance calculations
        // void computeCovariances();
        // const Eigen::MatrixXd & getDenseBlockCovariance(int di1, int di2);
        // Eigen::MatrixXd getSparseSparseCovariance(int si1, int si2);
        // Eigen::MatrixXd getDenseSparseCovariance(int di, int si);
        .def("computeCovariances", &Optimizer2::computeCovariances)
        .def("computeDiagonalCovariances", &Optimizer2::computeDiagonalCovariances)
        
        /// \brief Evaluate the error at the current state.
        .def("evaluateError", &Optimizer2::evaluateError)

        /// \brief Get dense design variable i.
        .def("densignVariable", &Optimizer2::designVariable, return_internal_reference<>())

        /// \brief how many dense design variables are involved in the problem
        .def("numDesignVariables", &Optimizer2::numDesignVariables)


        .def("printTiming", &Optimizer2::printTiming)
        .def("computeHessian", &Optimizer2::computeHessian)
   
        ;


    class_<OptimizerRpropOptions>("OptimizerRpropOptions", init<>())
        .def_readwrite("etaMinus",&OptimizerRpropOptions::etaMinus)
        .def_readwrite("etaPlus",&OptimizerRpropOptions::etaPlus)
        .def_readwrite("initialDelta",&OptimizerRpropOptions::initialDelta)
        .def_readwrite("minDelta",&OptimizerRpropOptions::minDelta)
        .def_readwrite("maxDelta",&OptimizerRpropOptions::maxDelta)
        .def_readwrite("maxIterations",&OptimizerRpropOptions::maxIterations)
        .def_readwrite("verbose",&OptimizerRpropOptions::verbose)
        .def_readwrite("nThreads", &OptimizerRpropOptions::nThreads)
        ;

    class_<OptimizerRprop, boost::shared_ptr<OptimizerRprop> >("OptimizerRprop", init<>("OptimizerRprop(): Constructor with default options"))

        .def(init<const OptimizerRpropOptions&>("OptimizerRprop(OptimizerRpropOptions options): Constructor with custom options"))
        .def(init<const sm::PropertyTree&>("OptimizerRprop(PropertyTree propertyTree): Constructor from sm::PropertyTree"))

        .def("setProblem", (void (OptimizerRprop::*)(boost::shared_ptr<OptimizationProblemBase>))&OptimizerRprop::setProblem, "Set up to work on the optimization problem.")
        .def("checkProblemSetup", (void (OptimizerRprop::*)(void))&OptimizerRprop::checkProblemSetup,
             "Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is hooked up to design variables and running "
             "finite differences on error terms where this is possible.")
//
        .def("initialize", &OptimizerRprop::initialize,
             "Initialize the optimizer to run on an optimization problem. optimize() will call initialize() upon the first call.")

        .def("optimize", &OptimizerRprop::optimize,
             "Run the optimization")
        .add_property("options", make_function(&OptimizerRprop::options, return_internal_reference<>()),
                      "The optimizer options.")

        .add_property("gradientNorm", &OptimizerRprop::getGradientNorm,
                      "The norm of the gradient of the objective function.")

        .add_property("numberOfIterations", &OptimizerRprop::getNumberOfIterations,
                      "Get the number of iterations the solver has run. If it has never been started, the value will be zero.")
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerRprop>, boost::shared_ptr<const OptimizerRprop> >();

}

