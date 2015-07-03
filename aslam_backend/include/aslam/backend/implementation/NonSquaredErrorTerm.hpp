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

  } // namespace backend
} // namespace aslam
