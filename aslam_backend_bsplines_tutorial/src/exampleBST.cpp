#include <aslam/backend/ErrorTermObservationBST.hpp>
#include <aslam/backend/ErrorTermMotionBST.hpp>
#include <aslam/backend/ErrorTermPriorBST.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/Optimizer2Options.hpp>
#include <aslam/backend/DenseQrLinearSystemSolver.hpp>
#include <iostream>
// Bring in some random number generation from Schweizer Messer.
#include <sm/random.hpp>
#include <vector>
#include <algorithm>
#include <boost/foreach.hpp>

#include <bsplines/BSplinePose.hpp>
//#include <aslam/splines/BSplinePoseDesignVariable.hpp>
#include <aslam/backend/EuclideanPoint.hpp>
#include <sm/kinematics/RotationVector.hpp>
#include <aslam/splines/OPTBSpline.hpp>
#include <aslam/splines/implementation/OPTBSplineImpl.hpp>
#include <bsplines/EuclideanBSpline.hpp>
#include <aslam/backend/Scalar.hpp>

template <typename TConf, int ISplineOrder, int IDim, bool BDimRequired> struct ConfCreator {
	static inline TConf create(){
		return TConf(typename TConf::ManifoldConf(IDim), ISplineOrder);
	}
};

template <typename TConf, int ISplineOrder, int IDim> struct ConfCreator<TConf, ISplineOrder, IDim, false> {
	static inline TConf create(){
		BOOST_STATIC_ASSERT_MSG(IDim == TConf::Dimension::VALUE, "impossible dimension selected!");
		return TConf(typename TConf::ManifoldConf(), ISplineOrder);
	}
};

template <typename TConf, int ISplineOrder, int IDim> inline TConf createConf(){
	return ConfCreator<TConf, ISplineOrder, IDim, TConf::Dimension::IS_DYNAMIC>::create();
}


int main(int argc, char ** argv)
{
  if(argc != 2)
    {
      std::cout << "Usage: example K\n";
      std::cout << "The argument K is the number of timesteps to include in the optimization\n";
      return 1;
    }

  const int K = atoi(argv[1]);

  try 
    {
      // The true wall position
      const double true_w = 5.0;
      
      // The noise properties.
      const double sigma_n = 0.01;
      const double sigma_u = 0.1;
      const double sigma_x = 0.01;

      // Create random odometry
      std::vector<double> true_u_k(K);
      BOOST_FOREACH(double & u, true_u_k)
	{
	  u = 1;//sm::random::uniform();
	}
      
      // Create the noisy odometry
      std::vector<double> u_k(K);
      for(int k = 0; k < K; ++k)
	{
	  u_k[k] = true_u_k[k] + sigma_u * sm::random::normal();
	}

      // Create the states from noisy odometry.
      std::vector<double> x_k(K);
      std::vector<double> true_x_k(K);
      x_k[0] = 10.0;
      true_x_k[0] = 10.0;
      for(int k = 1; k < K; ++k)
	{
	  true_x_k[k] = true_x_k[k-1] + true_u_k[k];
	  x_k[k] = x_k[k-1] + u_k[k];
	}


      // Create the noisy measurments
      std::vector<double> y_k(K);
      for(int k = 0; k < K; ++k)
	{
	  //y_k[k] = (1.0 / (true_w - true_x_k[k])) + sigma_n * sm::random::normal();
	  y_k[k] = (true_w / true_x_k[k]) + sigma_n * sm::random::normal();
	  //std::cout << "y_" << k << " is: " << y_k[k] << std::endl;
	}
      
      // Now we can build an optimization problem.
      boost::shared_ptr<aslam::backend::OptimizationProblem> problem( new aslam::backend::OptimizationProblem);


		// init BSPline
//		int splineOder = 4;
//		boost::shared_ptr<sm::kinematics::RotationVector> rotVec(new sm::kinematics::RotationVector());
//		bsplines::BSplinePose pose(splineOder, rotVec);
//		Eigen::Matrix4d initPose = Eigen::Matrix4d::Identity();
//		pose.initPoseSpline(0, K, initPose, initPose);

		aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline robotPosSpline(createConf<bsplines::EuclideanBSpline<4, 1>::CONF, 4, 1>());
		const int pointSize = robotPosSpline.getPointSize();

		typename aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::point_t initPoint(pointSize);
		typename aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::point_t p1(pointSize);

		std::cout << "Init point is: " << std::endl << initPoint << std::endl;


		//robotPosSpline.getManifold().randomizePoint(initPoint);

		initPoint(0,0) = x_k[0];
		p1(0,0) = 50;

		std::cout << "Init point is: " << std::endl << initPoint << std::endl;
		robotPosSpline.initConstantUniformSpline(0, K, 10*K, initPoint);

		robotPosSpline.addControlVertex(50, p1);

//		aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecPosExpr0 = robotPosSpline.getExpressionFactoryAt<1>(50).getValueExpression(0);
//		double v = vecPosExpr0.evaluate()(0);

		std::cout << "Init point is: " << std::endl << initPoint << std::endl;
		//std::cout << "v is: " << std::endl << v << std::endl;
		// add the robot pose spline to the problem
		for(size_t i = 0; i < robotPosSpline.numDesignVariables(); i++)
		{
			robotPosSpline.designVariable(i)->setActive(true);
			problem->addDesignVariable(robotPosSpline.designVariable(i), false);
		}

		//const int pointSize = robotPosSpline.getPointSize();

		//aslam::splines::BSplinePoseDesignVariable bSplinePoseDV(pose);

		// set up wall position
		double wallPosition = true_w + sm::random::normal();

      // First, create a design variable for the wall position.
      boost::shared_ptr<aslam::backend::Scalar> dv_w(new aslam::backend::Scalar(wallPosition));
      // Setting this active means we estimate it.
      dv_w->setActive(true);
      // Add it to the optimization problem.
      //problem->addDesignVariable(dv_w);

      // Now create a prior for this initial state.
      aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecPosExpr = robotPosSpline.getExpressionFactoryAt<1>(0).getValueExpression(1);
      boost::shared_ptr<aslam::backend::ErrorTermPriorBST> prior(new aslam::backend::ErrorTermPriorBST(vecPosExpr, true_x_k[0], sigma_x * sigma_x));
//      // and add it to the problem.
      problem->addErrorTerm(prior);

      // Now march through the states creating design variables,
      // odometry error terms and measurement error terms.
      for(int k = 1; k < K; ++k)
		{
		  // Create odometry error
          aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecVelExpr = robotPosSpline.getExpressionFactoryAt<1>(k).getValueExpression(0);
          std::cout << "vel input is: " << u_k[k] << std::endl;
		  boost::shared_ptr<aslam::backend::ErrorTermMotionBST> em(new aslam::backend::ErrorTermMotionBST(vecVelExpr, u_k[k], sigma_u * sigma_u));
		  problem->addErrorTerm(em);

		  // Create observation error
//	      aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecPosExpr = robotPosSpline.getExpressionFactoryAt<1>(k).getValueExpression(0);
//		  boost::shared_ptr<aslam::backend::ErrorTermObservationBST> eo(new aslam::backend::ErrorTermObservationBST(vecPosExpr, dv_w,  y_k[k], sigma_n * sigma_n));
//		  problem->addErrorTerm(eo);
		}

      // Now we have a valid optimization problem full of design variables and error terms.
      // Create some optimization options.
      aslam::backend::Optimizer2Options options;
      options.verbose = true;
      options.linearSystemSolver.reset(new DenseQrLinearSystemSolver());
//      options.levenbergMarquardtLambdaInit = 10;
      options.doSchurComplement = false;
//      options.doLevenbergMarquardt = true;
      // Force it to over-optimize
      options.convergenceDeltaX = 1e-12;
      options.convergenceDeltaJ = 1e-12;
      // Then create the optimizer and go!
      aslam::backend::Optimizer2 optimizer(options);
      optimizer.setProblem( problem );

      // the wall is at
      std::cout << "Before optimization, the wall is at: " << std::endl << dv_w->toExpression().toScalar() << std::endl;

      optimizer.optimize();

      // the wall is at
      std::cout << "After optimization, the wall is at: " << std::endl << dv_w->toExpression().toScalar() << std::endl;

      for(int i = 0; i < K; i += 10)
      {
    	  aslam::splines::OPTBSpline<bsplines::EuclideanBSpline<4, 1>::CONF>::BSpline::expression_t vecPosExpr = robotPosSpline.getExpressionFactoryAt<1>(i).getValueExpression(0);
          std::cout << "Robot at " << i << " is: " << std::endl << vecPosExpr.evaluate()(0) << std::endl;
      }
    }
  catch(const std::exception & e)
    {
      std::cout << "Exception during processing: " << e.what();
      return 1;
    }

  std::cout << "Processing completed successfully\n";
  return 0;
}
