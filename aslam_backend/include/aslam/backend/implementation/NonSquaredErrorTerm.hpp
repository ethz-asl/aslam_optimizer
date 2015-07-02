#include <sm/eigen/assert_macros.hpp>

namespace aslam {
  namespace backend {

    template <typename MEstimatorType>
    boost::shared_ptr<MEstimatorType> NonSquaredErrorTerm::getMEstimatorPolicy() {
      return boost::dynamic_pointer_cast<MEstimatorType>(_mEstimatorPolicy);
    }

    template<typename ITERATOR_T>
    void NonSquaredErrorTerm::setDesignVariablesIterator(ITERATOR_T start, ITERATOR_T end)
    {
      /// \todo Set the back link to the error term in the design variable.
      SM_ASSERT_EQ(aslam::UnsupportedOperationException, _designVariables.size(), 0, "The design variable container already has objects. The design variables may only be set once");
      /// \todo: set the back-link in the design variable.
      int ii = 0;
      for (ITERATOR_T i = start; i != end; ++i, ++ii) {
        SM_ASSERT_TRUE(aslam::InvalidArgumentException, *i != NULL, "Design variable " << ii << " is null");
      }
      _designVariables.insert(_designVariables.begin(), start, end);
    }

    double NonSquaredErrorTerm::getWeight() const
    {
      return _w;
    }

    void NonSquaredErrorTerm::setWeight(const double w)
    {
      _w = w;
    }

    void NonSquaredErrorTerm::setError(const double e)
    {
      _error = e;
    }

    void NonSquaredErrorTerm::getWeightedJacobians(JacobianContainer& outJc, bool useMEstimator)
    {
      // take a copy. \todo Don't take a copy.
      evaluateJacobians(outJc);
      Eigen::Matrix<double, 1, 1> w;
      w << _w;
      outJc.applyChainRule(w);
      JacobianContainer::map_t::iterator it = outJc.begin();
      double sqrtWeight = 1.0;
      if (useMEstimator)
        sqrtWeight = sqrt(_mEstimatorPolicy->getWeight(getRawError()));
      for (; it != outJc.end(); ++it) {
        it->second *=  sqrtWeight * it->first->scaling();
      }
    }

    double NonSquaredErrorTerm::getWeightedError(bool useMEstimator) const
    {
      double mEstWeight = 1.0;
      if (useMEstimator) {
        SM_ASSERT_TRUE(Exception, _mEstimatorPolicy != nullptr, "");
        mEstWeight = sqrt(_mEstimatorPolicy->getWeight(getRawError()));
      }
      return _error * mEstWeight;
    }

    void NonSquaredErrorTerm::checkJacobiansFinite() const {
//      JacobianContainer J(C);
//      evaluateJacobians(J);
//      for (JacobianContainer::map_t::iterator it = J.begin(); it != J.end(); ++it) {
//        SM_ASSERT_MAT_IS_FINITE(Exception, it->second,
//                                "Jacobian is not finite!");
//      }
    }

    void NonSquaredErrorTerm::checkJacobiansNumerical(double tolerance) {
//
//      JacobianContainer J(C);
//      evaluateJacobians(J);
//
//
//      detail::ErrorTermFsFunctor<C> functor(*this);
//      sm::eigen::NumericalDiff<detail::ErrorTermFsFunctor<C> >
//        numdiff(functor, tolerance);
//      int inputSize = 0;
//      for (size_t i = 0; i < numDesignVariables(); i++) {
//        inputSize += designVariable(i)->minimalDimensions();
//      }
//      const Eigen::MatrixXd JNumComp =
//        numdiff.estimateJacobian(Eigen::VectorXd::Zero(inputSize));
//      int offset = 0;
//      for (size_t i = 0; i < numDesignVariables(); i++) {
//        DesignVariable* d = designVariable(i);
//        const Eigen::MatrixXd JAna = J.Jacobian(d);
//        const Eigen::MatrixXd JNum =
//          JNumComp.block(0, offset, C, d->minimalDimensions());
//        for (int r = 0; r < JAna.rows(); ++r)
//          for (int c = 0; c < JAna.cols(); ++c)
//            SM_ASSERT_NEAR(Exception, JAna(r, c), JNum(r, c), tolerance,
//            "Analytical and numerical Jacobians differ!");
//        offset += d->minimalDimensions();
//      }
    }

  } // namespace backend
} // namespace aslam
