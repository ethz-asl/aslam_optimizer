#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/ScalarNonSquaredErrorTerm.hpp>
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


    enum_<OptimizerRpropOptions::Method>("RpropMethod")
        .value("RPROP_PLUS", OptimizerRpropOptions::Method::RPROP_PLUS)
        .value("RPROP_MINUS", OptimizerRpropOptions::Method::RPROP_MINUS)
        .value("IRPROP_MINUS", OptimizerRpropOptions::Method::IRPROP_MINUS)
        .value("IRPROP_PLUS", OptimizerRpropOptions::Method::IRPROP_PLUS)
        ;

    class_<OptimizerRpropOptions>("OptimizerRpropOptions", init<>())
        .def_readwrite("etaMinus",&OptimizerRpropOptions::etaMinus)
        .def_readwrite("etaPlus",&OptimizerRpropOptions::etaPlus)
        .def_readwrite("initialDelta",&OptimizerRpropOptions::initialDelta)
        .def_readwrite("minDelta",&OptimizerRpropOptions::minDelta)
        .def_readwrite("maxDelta",&OptimizerRpropOptions::maxDelta)
        .def_readwrite("maxIterations",&OptimizerRpropOptions::maxIterations)
        .def_readwrite("nThreads", &OptimizerRpropOptions::nThreads)
        .def_readwrite("convergenceGradientNorm", &OptimizerRpropOptions::convergenceGradientNorm)
        .def_readwrite("convergenceDx", &OptimizerRpropOptions::convergenceDx)
        .def_readwrite("convergenceDObjective", &OptimizerRpropOptions::convergenceDObjective)
        .def_readwrite("regularizer", &OptimizerRpropOptions::regularizer)
        .def_readwrite("method", &OptimizerRpropOptions::method)
        ;

    enum_<RpropReturnValue::ConvergenceCriterion>("RpropConvergenceCriterion")
        .value("IN_PROGRESS", RpropReturnValue::ConvergenceCriterion::IN_PROGRESS)
        .value("FAILURE", RpropReturnValue::ConvergenceCriterion::FAILURE)
        .value("GRADIENT_NORM", RpropReturnValue::ConvergenceCriterion::GRADIENT_NORM)
        .value("DX", RpropReturnValue::ConvergenceCriterion::DX)
        .value("DOBJECTIVE", RpropReturnValue::ConvergenceCriterion::DOBJECTIVE)
        ;

    class_<RpropReturnValue>("RpropReturnValue", init<>())
        .def_readwrite("convergence",&RpropReturnValue::convergence)
        .def_readwrite("nIterations",&RpropReturnValue::nIterations)
        .def_readwrite("nGradEvaluations",&RpropReturnValue::nGradEvaluations)
        .def_readwrite("nObjectiveEvaluations",&RpropReturnValue::nObjectiveEvaluations)
        .def_readwrite("gradientNorm",&RpropReturnValue::gradientNorm)
        .def_readwrite("maxDx",&RpropReturnValue::maxDx)
        .def_readwrite("error",&RpropReturnValue::error)
        .def_readwrite("derror",&RpropReturnValue::derror)
        .def("success", &RpropReturnValue::success)
        .def("failure", &RpropReturnValue::failure)
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

       .def("reset", &OptimizerRprop::reset,
            "Reset internal states but don't re-initialize the whole problem")

        .def("optimize", make_function(&OptimizerRprop::optimize, return_internal_reference<>()),
             "Run the optimization")
        .add_property("status", make_function(&OptimizerRprop::getStatus, return_internal_reference<>()),
                      "Return the status")
        .add_property("options", make_function(&OptimizerRprop::options, return_internal_reference<>()),
                      "The optimizer options.")

        .add_property("gradientNorm", &OptimizerRprop::getGradientNorm,
                      "The norm of the gradient of the objective function.")

        .add_property("numberOfIterations", &OptimizerRprop::getNumberOfIterations,
                      "Get the number of iterations the solver has run. If it has never been started, the value will be zero.")
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerRprop>, boost::shared_ptr<const OptimizerRprop> >();


    enum_<BFGSReturnValue::ConvergenceCriterion>("BFGSConvergenceCriterion")
        .value("IN_PROGRESS", BFGSReturnValue::ConvergenceCriterion::IN_PROGRESS)
        .value("FAILURE", BFGSReturnValue::ConvergenceCriterion::FAILURE)
        .value("GRADIENT_NORM", BFGSReturnValue::ConvergenceCriterion::GRADIENT_NORM)
        .value("DX", BFGSReturnValue::ConvergenceCriterion::DX)
        .value("DOBJECTIVE", BFGSReturnValue::ConvergenceCriterion::DOBJECTIVE)
        ;

    class_<LineSearchOptions>("LineSearchOptions", init<>())
        .def_readwrite("nThreadsError",&LineSearchOptions::nThreadsError)
        .def_readwrite("nThreadsGradient",&LineSearchOptions::nThreadsGradient)
        .def_readwrite("c1WolfeCondition",&LineSearchOptions::c1WolfeCondition)
        .def_readwrite("c2WolfeCondition", &LineSearchOptions::c2WolfeCondition)
        .def_readwrite("maxStepLength", &LineSearchOptions::maxStepLength)
        .def_readwrite("minStepLength", &LineSearchOptions::minStepLength)
        .def_readwrite("xtol", &LineSearchOptions::xtol)
        .def_readwrite("initialStepLength", &LineSearchOptions::initialStepLength)
        .def_readwrite("nMaxIterWolfe1", &LineSearchOptions::nMaxIterWolfe1)
        .def_readwrite("nMaxIterWolfe2", &LineSearchOptions::nMaxIterWolfe2)
        .def_readwrite("nMaxIterZoom", &LineSearchOptions::nMaxIterZoom)
        ;

    class_<OptimizerBFGSOptions>("OptimizerBFGSOptions", init<>())
        .def_readwrite("linesearch",&OptimizerBFGSOptions::linesearch)
        .def_readwrite("maxIterations",&OptimizerBFGSOptions::maxIterations)
        .def_readwrite("convergenceGradientNorm", &OptimizerBFGSOptions::convergenceGradientNorm)
        .def_readwrite("convergenceDx", &OptimizerBFGSOptions::convergenceDx)
        .def_readwrite("convergenceDObjective", &OptimizerBFGSOptions::convergenceDObjective)
        .def_readwrite("regularizer", &OptimizerBFGSOptions::regularizer)
        ;

    class_<BFGSReturnValue>("BFGSReturnValue", init<>())
        .def_readwrite("convergence",&BFGSReturnValue::convergence)
        .def_readwrite("nIterations",&BFGSReturnValue::nIterations)
        .def_readwrite("nGradEvaluations",&BFGSReturnValue::nGradEvaluations)
        .def_readwrite("nObjectiveEvaluations",&BFGSReturnValue::nObjectiveEvaluations)
        .def_readwrite("gradientNorm",&BFGSReturnValue::gradientNorm)
        .def_readwrite("error",&BFGSReturnValue::error)
        .def_readwrite("derror",&BFGSReturnValue::derror)
        .def_readwrite("maxDx",&BFGSReturnValue::maxDx)
        .def("reset", &BFGSReturnValue::reset)
        .def("success", &BFGSReturnValue::success)
        .def("failure", &BFGSReturnValue::failure)
        ;

    class_<OptimizerBFGS, boost::shared_ptr<OptimizerBFGS> >("OptimizerBFGS", init<>("OptimizerBFGS(): Constructor with default options"))

        .def(init<const OptimizerBFGSOptions&>("OptimizerBFGSOptions(OptimizerRpropOptions options): Constructor with custom options"))
        .def(init<const sm::PropertyTree&>("OptimizerRprop(PropertyTree propertyTree): Constructor from sm::PropertyTree"))

        .def("setProblem", (void (OptimizerBFGS::*)(boost::shared_ptr<OptimizationProblemBase>))&OptimizerBFGS::setProblem, "Set up to work on the optimization problem.")
        .def("checkProblemSetup", (void (OptimizerBFGS::*)(void))&OptimizerBFGS::checkProblemSetup,
             "Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is hooked up to design variables and running "
             "finite differences on error terms where this is possible.")
//
        .def("initialize", &OptimizerBFGS::initialize,
             "Initialize the optimizer to run on an optimization problem. optimize() will call initialize() upon the first call.")

        .def("reset", &OptimizerBFGS::reset,
             "Reset internal states but don't re-initialize the whole problem")

        .def("optimize", make_function(&OptimizerBFGS::optimize, return_internal_reference<>()),
             "Run the optimization")
        .add_property("status", make_function(&OptimizerBFGS::getStatus, return_internal_reference<>()),
                      "Return the status")
        .add_property("options", make_function(&OptimizerBFGS::getOptions, return_internal_reference<>()), &OptimizerBFGS::setOptions,
                      "The optimizer options.")

        .add_property("gradientNorm", &OptimizerBFGS::getGradientNorm,
                      "The norm of the gradient of the objective function.")

        .add_property("numberOfIterations", &OptimizerBFGS::getNumberOfIterations,
                      "Get the number of iterations the solver has run. If it has never been started, the value will be zero.")
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerBFGS>, boost::shared_ptr<const OptimizerBFGS> >();

}

