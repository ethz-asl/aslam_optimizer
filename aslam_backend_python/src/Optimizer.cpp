#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy_eigen/boost_python_headers.hpp>
#include <aslam/backend/OptimizerBase.hpp>
#include <aslam/backend/util/OptimizerProblemManagerBase.hpp>
#include <aslam/backend/OptimizerCallbackManager.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/OptimizerRprop.hpp>
#include <aslam/backend/OptimizerBFGS.hpp>
#include <aslam/backend/OptimizerSgd.hpp>
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

template <typename T>
std::string toString(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

void addCallbackWrapper(aslam::backend::callback::Registry& registry,
                        aslam::backend::callback::Occasion occasion,
                        boost::python::object callback) {
  registry.add(occasion, [callback]() {
    callback();
  });
}

void addCallbackWithArgWrapper(aslam::backend::callback::Registry& registry,
                        aslam::backend::callback::Occasion occasion,
                        boost::python::object callback) {
  registry.add(occasion, [callback](const aslam::backend::callback::Argument& arg) {
    callback(arg);
  });
}

boost::python::list processedErrorTermsWrapper(const aslam::backend::OptimizerStatusSgd& status) {
  boost::python::list l;
  for (const auto& item : status.processedErrorTerms) {
    l.append(item);
  }
  return l;
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

    enum_<callback::Occasion>("CallbackOccasion")
        .value("COST_UPDATED", callback::Occasion::COST_UPDATED)
        .value("DESIGN_VARIABLE_UPDATE_COMPUTED", callback::Occasion::DESIGN_VARIABLE_UPDATE_COMPUTED)
        .value("DESIGN_VARIABLES_UPDATED", callback::Occasion::DESIGN_VARIABLES_UPDATED)
        .value("LINEAR_SYSTEM_SOLVED", callback::Occasion::LINEAR_SYSTEM_SOLVED)
        .value("OPTIMIZATION_INITIALIZED", callback::Occasion::OPTIMIZATION_INITIALIZED)
        .value("ITERATION_START", callback::Occasion::ITERATION_START)
        .value("ITERATION_END", callback::Occasion::ITERATION_END)
        .value("OPTIMIZATION_INITIALIZED", callback::Occasion::OPTIMIZATION_INITIALIZED)
        .value("RESIDUALS_UPDATED", callback::Occasion::RESIDUALS_UPDATED)
    ;

    class_<callback::Registry>("CallbackRegistry", no_init)
        .def("clear", (void (callback::Registry::*)(void))&callback::Registry::clear, "Removes all callbacks")
        .def("clear", (void (callback::Registry::*)(callback::Occasion))&callback::Registry::clear, "Removes all callbacks for a specific occasion")
        .def("add", &addCallbackWrapper, "Adds a callback for a specific occasion")
        .def("addWithArg", &addCallbackWithArgWrapper, "Adds a callback for a specific occasion that gets a callback argument passed")
        .def("numCallbacks", &callback::Registry::numCallbacks, "Number of callbacks for a specific occasion")
    ;

    class_<callback::Argument>("CallbackArgument", no_init)
        .def_readonly("occasion", &callback::Argument::occasion)
        .def_readonly("currentCost", &callback::Argument::currentCost)
        .def_readonly("previousLowestCost", &callback::Argument::previousLowestCost)
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

    class_<OptimizerOptionsBase, boost::shared_ptr<OptimizerOptionsBase> >("OptimizerOptionsBase", init<>("OptimizerOptionsBase(): Constructor"))
        .def(init<const sm::PropertyTree&>("OptimizerOptionsBase(sm::PropertyTree pt): Constructor"))
        .def_readwrite("convergenceGradientNorm", &OptimizerOptionsBase::convergenceGradientNorm)
        .def_readwrite("convergenceDeltaX", &OptimizerOptionsBase::convergenceDeltaX)
        .def_readwrite("convergenceDeltaObjective", &OptimizerOptionsBase::convergenceDeltaObjective)
        .def_readwrite("maxIterations",&OptimizerOptionsBase::maxIterations)
        .def_readwrite("numThreadsGradient", &OptimizerOptionsBase::numThreadsGradient)
        .def_readwrite("numThreadsError", &OptimizerOptionsBase::numThreadsError)
        .def("__str__", &toString<OptimizerOptionsBase>)
        ;

    enum_<ConvergenceStatus>("ConvergenceStatus")
        .value("IN_PROGRESS", ConvergenceStatus::IN_PROGRESS)
        .value("FAILURE", ConvergenceStatus::FAILURE)
        .value("GRADIENT_NORM", ConvergenceStatus::GRADIENT_NORM)
        .value("DX", ConvergenceStatus::DX)
        .value("DOBJECTIVE", ConvergenceStatus::DOBJECTIVE)
        ;

    class_<OptimizerStatus, boost::shared_ptr<OptimizerStatus> >("OptimizerStatus")
        .def_readwrite("convergence",&OptimizerStatus::convergence)
        .def_readwrite("numIterations",&OptimizerStatus::numIterations)
        .def_readwrite("numDerivativeEvaluations",&OptimizerStatus::numDerivativeEvaluations)
        .def_readwrite("numObjectiveEvaluations",&OptimizerStatus::numObjectiveEvaluations)
        .def_readwrite("gradientNorm",&OptimizerStatus::gradientNorm)
        .def_readwrite("maxDeltaX",&OptimizerStatus::maxDeltaX)
        .def_readwrite("error",&OptimizerStatus::error)
        .def_readwrite("deltaError",&OptimizerStatus::deltaError)
        .def("success", &OptimizerStatus::success)
        .def("failure", &OptimizerStatus::failure)
        .def("__str__", &toString<OptimizerStatus>)
        ;

    class_<OptimizerBase, boost::shared_ptr<OptimizerBase>, boost::noncopyable >("OptimizerBase", no_init)

        .def("setProblem", pure_virtual(&OptimizerBase::setProblem),
             "Set up to work on the optimization problem.")

        .def("checkProblemSetup", pure_virtual(&OptimizerBase::checkProblemSetup),
             "Do a bunch of checks to see if the problem is well-defined. This includes checking that every error term is hooked up to design variables and running "
             "finite differences on error terms where this is possible.")

        .def("initialize", &OptimizerBase::initialize,
             "Initialize the optimizer to run on an optimization problem. optimize() will call initialize() upon the first call.")

        .def("isInitialized", pure_virtual(&OptimizerBase::isInitialized),
             "Initialize the optimizer to run on an optimization problem. optimize() will call initialize() upon the first call.")

        .def("reset", &OptimizerBase::reset,
             "Reset internal states but don't re-initialize the whole problem")

        .def("optimize", pure_virtual(&OptimizerBase::optimize),
             "Run the optimization")

        .add_property("status", make_function(&OptimizerBase::getStatus, return_internal_reference<>()),
                      "Status of the optimizer")

        .add_property("options", make_function(&OptimizerBase::getOptions, return_internal_reference<>()), &OptimizerBase::setOptions,
                      "Options of the optimizer")

        .def("isConverged", &OptimizerBase::isConverged,
             "Has the optimizer converged?")

        .def("isFailed", &OptimizerBase::isFailed,
            "Has the optimizer failed?")

        .def("isInProgress", &OptimizerBase::isInProgress,
             "Is the optimizer still running?")

        .add_property("callback", make_function(&OptimizerBase::callback, return_internal_reference<>()),
                      "Callback manager")
        ;

    class_<OptimizerProblemManagerBase, boost::shared_ptr<OptimizerProblemManagerBase>, bases<OptimizerBase>, boost::noncopyable >("OptimizerProblemManagerBase", no_init)
        ;

    enum_<OptimizerOptionsRprop::Method>("RpropMethod")
        .value("RPROP_PLUS", OptimizerOptionsRprop::Method::RPROP_PLUS)
        .value("RPROP_MINUS", OptimizerOptionsRprop::Method::RPROP_MINUS)
        .value("IRPROP_MINUS", OptimizerOptionsRprop::Method::IRPROP_MINUS)
        .value("IRPROP_PLUS", OptimizerOptionsRprop::Method::IRPROP_PLUS)
        ;

    class_<OptimizerOptionsRprop, boost::shared_ptr<OptimizerOptionsRprop>, bases<OptimizerOptionsBase> >("OptimizerOptionsRprop", init<>())
        .def_readwrite("etaMinus",&OptimizerOptionsRprop::etaMinus)
        .def_readwrite("etaPlus",&OptimizerOptionsRprop::etaPlus)
        .def_readwrite("initialDelta",&OptimizerOptionsRprop::initialDelta)
        .def_readwrite("minDelta",&OptimizerOptionsRprop::minDelta)
        .def_readwrite("maxDelta",&OptimizerOptionsRprop::maxDelta)
        .def_readwrite("regularizer", &OptimizerOptionsRprop::regularizer)
        .def_readwrite("method", &OptimizerOptionsRprop::method)
        .def_readwrite("useDenseJacobianContainer", &OptimizerOptionsRprop::useDenseJacobianContainer)
        .def("__str__", &toString<OptimizerOptionsRprop>)
        ;


    class_<OptimizerRprop, boost::shared_ptr<OptimizerRprop>, bases<OptimizerProblemManagerBase> >("OptimizerRprop", init<>("OptimizerRprop(): Constructor with default options"))
        .def(init<const OptimizerOptionsRprop&>("OptimizerRprop(OptimizerOptionsRprop options): Constructor with custom options"))
        .def(init<const sm::PropertyTree&>("OptimizerRprop(PropertyTree propertyTree): Constructor from sm::PropertyTree"))
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerRprop>, boost::shared_ptr<const OptimizerRprop> >();


    class_<LineSearchOptions, boost::shared_ptr<LineSearchOptions> >("LineSearchOptions", init<>())
        .def_readwrite("c1WolfeCondition",&LineSearchOptions::c1WolfeCondition)
        .def_readwrite("c2WolfeCondition", &LineSearchOptions::c2WolfeCondition)
        .def_readwrite("maxStepLength", &LineSearchOptions::maxStepLength)
        .def_readwrite("minStepLength", &LineSearchOptions::minStepLength)
        .def_readwrite("xtol", &LineSearchOptions::xtol)
        .def_readwrite("initialStepLength", &LineSearchOptions::initialStepLength)
        .def_readwrite("nMaxIterWolfe1", &LineSearchOptions::nMaxIterWolfe1)
        .def_readwrite("nMaxIterWolfe2", &LineSearchOptions::nMaxIterWolfe2)
        .def_readwrite("nMaxIterZoom", &LineSearchOptions::nMaxIterZoom)
        .def("__str__", &toString<LineSearchOptions>)
        ;

    class_<OptimizerOptionsBFGS, boost::shared_ptr<OptimizerOptionsBFGS>, bases<OptimizerOptionsBase> >("OptimizerOptionsBFGS", init<>())
        .def_readwrite("linesearch", &OptimizerOptionsBFGS::linesearch)
        .def_readwrite("useDenseJacobianContainer", &OptimizerOptionsBFGS::useDenseJacobianContainer)
        .def_readwrite("regularizer", &OptimizerOptionsBFGS::regularizer)
        .def("__str__", &toString<OptimizerOptionsBFGS>)
        ;

    class_<OptimizerBFGS, boost::shared_ptr<OptimizerBFGS>, bases<OptimizerProblemManagerBase> >("OptimizerBFGS", init<>("OptimizerBFGS(): Constructor with default options"))
        .def(init<const OptimizerOptionsBFGS&>("OptimizerBFGSOptions(OptimizerOptionsBFGS options): Constructor with custom options"))
        .def(init<const sm::PropertyTree&>("OptimizerRprop(PropertyTree propertyTree): Constructor from sm::PropertyTree"))
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerBFGS>, boost::shared_ptr<const OptimizerBFGS> >();


    class_<LearningRateScheduleBase, boost::shared_ptr<LearningRateScheduleBase>, boost::noncopyable>("__LearningRateScheduleBase", no_init)
        .def("computeDx", &LearningRateScheduleBase::computeDx)
        .def("initialize", &LearningRateScheduleBase::initialize)
        .def("check", &LearningRateScheduleBase::check)
        .add_property("name", make_function(&LearningRateScheduleBase::getName, return_value_policy<copy_const_reference>()))
        .add_property("currentRate", make_function(&LearningRateScheduleBase::getCurrentRate, return_value_policy<copy_const_reference>()))
        ;

    class_<LearningRateScheduleWithMomentumBase, boost::shared_ptr<LearningRateScheduleWithMomentumBase>, bases<LearningRateScheduleBase>, boost::noncopyable>("__LearningRateScheduleWithMomentumBase", no_init)
        .add_property("momentum", &LearningRateScheduleWithMomentumBase::getMomentum, &LearningRateScheduleWithMomentumBase::setMomentum)
        .add_property("prevDx", make_function(&LearningRateScheduleWithMomentumBase::getPrevDx, return_value_policy<copy_const_reference>()))
        ;

    class_<LearningRateScheduleConstant, boost::shared_ptr<LearningRateScheduleConstant>, bases<LearningRateScheduleWithMomentumBase> >("LearningRateScheduleConstant",
        init<const double, optional<const double> >("LearningRateScheduleConstant(rate, momentum): Constructor"))
        .def(init<const sm::PropertyTree&>("LearningRateScheduleConstant(PropertyTree pt): Constructor"))
        .add_property("lr", &LearningRateScheduleConstant::getLr, &LearningRateScheduleConstant::setLr)
        ;
    register_ptr_to_python< boost::shared_ptr<const LearningRateScheduleConstant> >();
    implicitly_convertible< boost::shared_ptr<LearningRateScheduleConstant>, boost::shared_ptr<const LearningRateScheduleConstant> >();

    class_<LearningRateScheduleOptimal, boost::shared_ptr<LearningRateScheduleOptimal>, bases<LearningRateScheduleWithMomentumBase> >("LearningRateScheduleOptimal",
        init<const double, const double, optional<const double> >("LearningRateScheduleOptimal(lr, tau, momentum): Constructor"))
        .def(init<const sm::PropertyTree&>("LearningRateScheduleOptimal(PropertyTree pt): Constructor"))
        .add_property("lr", &LearningRateScheduleOptimal::getLr, &LearningRateScheduleOptimal::setLr)
        .add_property("tau", &LearningRateScheduleOptimal::getTau, &LearningRateScheduleOptimal::setTau)
        ;
    register_ptr_to_python< boost::shared_ptr<const LearningRateScheduleOptimal> >();
    implicitly_convertible< boost::shared_ptr<LearningRateScheduleOptimal>, boost::shared_ptr<const LearningRateScheduleOptimal> >();

    class_<LearningRateScheduleRMSProp, boost::shared_ptr<LearningRateScheduleRMSProp>, bases<LearningRateScheduleWithMomentumBase> >("LearningRateScheduleRMSProp",
        init< const double, optional<const double, const double, const double, const bool, const double, const double, const double> >(
            "LearningRateScheduleRMSProp(lr, rho, epsilon, momentum, isStepAdapt, stepFactor, minLr, maxLr): Constructor"))
        .def(init<const sm::PropertyTree&>("LearningRateScheduleRMSProp(PropertyTree pt): Constructor"))
        .add_property("lr", &LearningRateScheduleRMSProp::getLr, &LearningRateScheduleRMSProp::setLr)
        .add_property("rho", &LearningRateScheduleRMSProp::getRho, &LearningRateScheduleRMSProp::setRho)
        .add_property("epsilon", &LearningRateScheduleRMSProp::getEpsilon, &LearningRateScheduleRMSProp::setEpsilon)
        .add_property("gradSqrAverage", make_function(&LearningRateScheduleRMSProp::getGradSqrAverage, return_value_policy<copy_const_reference>()))
        .add_property("alpha", make_function(&LearningRateScheduleRMSProp::getAlpha, return_value_policy<copy_const_reference>()))
        ;
    register_ptr_to_python< boost::shared_ptr<const LearningRateScheduleRMSProp> >();
    implicitly_convertible< boost::shared_ptr<LearningRateScheduleRMSProp>, boost::shared_ptr<const LearningRateScheduleRMSProp> >();

    class_<LearningRateScheduleAdaDelta, boost::shared_ptr<LearningRateScheduleAdaDelta>, bases<LearningRateScheduleBase> >("LearningRateScheduleAdaDelta",
        init< optional<const double, const double, const double> >("LearningRateScheduleAdaDelta(rho, epsilon, lr): Constructor"))
        .def(init<const sm::PropertyTree&>("LearningRateScheduleAdaDelta(PropertyTree pt): Constructor"))
        .add_property("lr", &LearningRateScheduleAdaDelta::getLr, &LearningRateScheduleAdaDelta::setLr)
        .add_property("rho", &LearningRateScheduleAdaDelta::getRho, &LearningRateScheduleAdaDelta::setRho)
        .add_property("epsilon", &LearningRateScheduleAdaDelta::getEpsilon, &LearningRateScheduleAdaDelta::setEpsilon)
        .add_property("gradSqrAverage", make_function(&LearningRateScheduleAdaDelta::getGradSqrAverage, return_value_policy<copy_const_reference>()))
        .add_property("dxSqrAverage", make_function(&LearningRateScheduleAdaDelta::getDxSqrAverage, return_value_policy<copy_const_reference>()))
        ;
    register_ptr_to_python< boost::shared_ptr<const LearningRateScheduleAdaDelta> >();
    implicitly_convertible< boost::shared_ptr<LearningRateScheduleAdaDelta>, boost::shared_ptr<const LearningRateScheduleAdaDelta> >();

    class_<LearningRateScheduleAdam, boost::shared_ptr<LearningRateScheduleAdam>, bases<LearningRateScheduleBase> >("LearningRateScheduleAdam",
        init< const double, optional<const double, const double, const double> >("LearningRateScheduleAdam(lr, rho1, rho2, epsilon): Constructor"))
        .def(init<const sm::PropertyTree&>("LearningRateScheduleAdam(PropertyTree pt): Constructor"))
        .add_property("lr", &LearningRateScheduleAdam::getLr, &LearningRateScheduleAdam::setLr)
        .add_property("rho1", &LearningRateScheduleAdam::getRho1)
        .add_property("rho2", &LearningRateScheduleAdam::getRho2)
        .add_property("epsilon", &LearningRateScheduleAdam::getEpsilon, &LearningRateScheduleAdam::setEpsilon)
        .add_property("gradAverage", make_function(&LearningRateScheduleAdam::getGradAverage, return_value_policy<copy_const_reference>()))
        .add_property("gradSqrAverage", make_function(&LearningRateScheduleAdam::getGradSqrAverage, return_value_policy<copy_const_reference>()))
        ;
    register_ptr_to_python< boost::shared_ptr<const LearningRateScheduleAdam> >();
    implicitly_convertible< boost::shared_ptr<LearningRateScheduleAdam>, boost::shared_ptr<const LearningRateScheduleAdam> >();


    class_<OptimizerOptionsSgd, bases<OptimizerOptionsBase> >("OptimizerOptionsSgd", init<>())
        .def_readwrite("batchSize", &OptimizerOptionsSgd::batchSize)
        .def_readwrite("useDenseJacobianContainer", &OptimizerOptionsSgd::useDenseJacobianContainer)
        .def_readwrite("regularizer", &OptimizerOptionsSgd::regularizer)
        .def_readwrite("learningRateSchedule", &OptimizerOptionsSgd::learningRateSchedule)
        .def("__str__", &toString<OptimizerOptionsSgd>)
        ;

    class_<OptimizerStatusSgd, bases<OptimizerStatus> >("OptimizerStatusSgd", init<>())
        .def_readonly("numBatches", &OptimizerStatusSgd::numBatches)
        .def_readonly("numSubIterations", &OptimizerStatusSgd::numSubIterations)
        .def_readonly("numTotalIterations", &OptimizerStatusSgd::numTotalIterations)
        .add_property("processedErrorTerms", &processedErrorTermsWrapper)
        .def("__str__", &toString<OptimizerStatusSgd>)
        ;

    class_<OptimizerSgd, boost::shared_ptr<OptimizerSgd>, bases<OptimizerProblemManagerBase> >("OptimizerSgd", init<>("OptimizerSgd(): Constructor with default options"))

        .def(init<const OptimizerSgd::Options&>("OptimizerSgd(OptimizerSgdOptions options): Constructor with custom options"))
        .def(init<const sm::PropertyTree&>("OptimizerSgd(PropertyTree propertyTree): Constructor from sm::PropertyTree"))

        .def("addBatch", &OptimizerSgd::addBatch< std::vector<ScalarNonSquaredErrorTerm*> >,
             "Add a batch of error terms")
        .def("addBatch", &OptimizerSgd::addBatch< std::vector< boost::shared_ptr<ScalarNonSquaredErrorTerm> > >,
             "Add a batch of error terms")
        .def("addBatch", &OptimizerSgd::addBatch< std::vector<ErrorTerm*> >,
            "Add a batch of error terms")
        .def("addBatch", &OptimizerSgd::addBatch< std::vector<boost::shared_ptr<ErrorTerm> > >,
            "Add a batch of error terms")

        .def("setCounters", &OptimizerSgd::setCounters,
             "Manually set the number of iterations. Use this, if you have already incorporated some training "
             "samples in another way (e.g. via batch optimization) in the beginning.")
        .def("setRandomSeed", &OptimizerSgd::setRandomSeed,
             "Set seed for random number generator used to shuffle data")
        ;
    implicitly_convertible< boost::shared_ptr<OptimizerSgd>, boost::shared_ptr<const OptimizerSgd> >();

    typedef std::vector< boost::shared_ptr<ScalarNonSquaredErrorTerm> > ScalarNonSquaredErrorTermBatch;
    class_<ScalarNonSquaredErrorTermBatch>("ScalarNonSquaredErrorTermBatch")
      .def(boost::python::vector_indexing_suite<ScalarNonSquaredErrorTermBatch>() )
      .def("__iter__", boost::python::iterator<ScalarNonSquaredErrorTermBatch>())
    ;
    typedef std::vector< boost::shared_ptr<ErrorTerm> > ErrorTermBatch;
    class_<ErrorTermBatch>("ErrorTermBatch")
      .def(boost::python::vector_indexing_suite<ErrorTermBatch>() )
      .def("__iter__", boost::python::iterator<ErrorTermBatch>())
    ;
}

