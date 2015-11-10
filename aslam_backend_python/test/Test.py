#!/usr/bin/env python
import aslam_backend as ab
import numpy as np

import unittest

class TestRprop(unittest.TestCase):
    def test_simple_optimization(self):
        options = ab.OptimizerRpropOptions()
        options.maxIterations = 500;
        options.nThreads = 1;
        options.convergenceGradientNorm = 1e-6
        options.method = ab.RpropMethod.RPROP_PLUS
        optimizer = ab.OptimizerRprop(options)
        problem = ab.OptimizationProblem()
        
        D = 2
        E = 3
        
        # Add some design variables.
        p2ds = [];
        for d in range(0, D):
          point = ab.Point2d(np.array([d, d]))
          p2ds.append(point);
          problem.addDesignVariable(point);
          point.setBlockIndex(d);
          point.setActive(True);
        
        # Add some error terms.
        errors = [];
        for e in range(0, E):
          for d in range(0, D):
            grad = np.array( [d+1, e+1] );
            err = ab.TestNonSquaredError(p2ds[d], grad);
            err._p = 1.0;
            errors.append(err);
            problem.addScalarNonSquaredErrorTerm(err);
            
        # Now let's optimize.
        optimizer.setProblem(problem);
        optimizer.checkProblemSetup();
        optimizer.optimize();
    
        self.assertLessEqual(optimizer.gradientNorm, 1e-3);
        
class TestMetropolisHastings(unittest.TestCase):
    def test_simple_sampling_problem(self):
        '''Sample from a twodimensional Gaussian distribution N(0, I)
        '''
      
        options = ab.SamplerMetropolisHastingsOptions()
        options.transitionKernelSigma = 0.1;
        sampler = ab.SamplerMetropolisHastings(options)
        negLogDensity = ab.OptimizationProblem()

        # Add a scalar design variables.
        point = ab.Point2d(np.array([0, 0]))
        point.setBlockIndex(0);
        point.setActive(True);
        negLogDensity.addDesignVariable(point);

        # Add the error term.
        grad = np.array( [-1., -1.] );
        err = ab.TestNonSquaredError(point, grad);
        err._p = 0.0; # set mean
        negLogDensity.addScalarNonSquaredErrorTerm(err);

        # Now let's sample.
        sampler.setNegativeLogDensity(negLogDensity);
        sampler.checkNegativeLogDensitySetup();
        sampler.run(10000);
        
if __name__ == '__main__':
    import rostest
    rostest.rosrun('aslam_backend_python', 'Rprop', TestRprop)
    rostest.rosrun('aslam_backend_python', 'MetropolisHastings', TestMetropolisHastings)