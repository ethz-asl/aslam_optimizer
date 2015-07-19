#!/usr/bin/env python
import aslam_backend as ab
import numpy as np

import unittest

class TestRprop(unittest.TestCase):
    def test_simple_optimization(self):
        options = ab.OptimizerRpropOptions()
        options.verbose = False;
        options.maxIterations = 500;
        options.nThreads = 1;
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
        
if __name__ == '__main__':
    import rostest
    rostest.rosrun('aslam_backend_python', 'Rprop', TestRprop)