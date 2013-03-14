#ifndef ASLAM_MESTIMATOR_POLICIES_HPP
#define ASLAM_MESTIMATOR_POLICIES_HPP

#include <cmath>
#include <string>
#include <sstream>
#include <array>

#include <aslam/Exceptions.hpp>
#include <sm/assert_macros.hpp>

namespace aslam {
  namespace backend {

    /// Inverse chi-squared cdf for p = 0.999 and df = 1...20
    constexpr std::array<double, 20> chi2invTable{{
      10.827566170662738,
      13.815510557964272,
      16.266236196238129,
      18.466826952903151,
      20.515005652432876,
      22.457744484825323,
      24.321886347856850,
      26.124481558376136,
      27.877164871256568,
      29.588298445074418,
      31.264133620239992,
      32.909490407360217,
      34.528178974870883,
      36.123273680398135,
      37.697298218353822,
      39.252354790768472,
      40.790216706902520,
      42.312396331679963,
      43.820195964517531,
      45.314746618125859}};

    class MEstimator {
    public:
      virtual ~MEstimator();
      virtual double getWeight(double squaredError) = 0;
      virtual std::string name() = 0;
    };


    class NoMEstimator : public MEstimator {
    public:
      virtual ~NoMEstimator();
      virtual double getWeight(double squaredError) {
        return 1.0;
      }
      virtual std::string name() {
        return "none";
      }
    };

    class  GemanMcClureMEstimator : public MEstimator {
    public:
      GemanMcClureMEstimator(double sigma2) : _sigma2(sigma2) {}
      virtual ~GemanMcClureMEstimator();

      virtual double getWeight(double error) {
        double se = _sigma2 + error;
        return (_sigma2) / (se * se);
      }
      virtual std::string name() {
        std::stringstream ss;
        ss << "Geman McClure (" << _sigma2 << ")";
        return ss.str();
      }

      double _sigma2;
    };

    class HuberMEstimator : public MEstimator {
    public:
      HuberMEstimator(double k) : _k(k), _k2(k* k) {}
      virtual ~HuberMEstimator();
      virtual double getWeight(double error) {
        return error < _k2 ? 1.0 : _k / sqrt(error);
      }
      virtual std::string name() {
        std::stringstream ss;
        ss << "Huber(" << _k << ")";
        return ss.str();
      }

      double _k;
      double _k2;
    };

    /** The class BlakeZissermanMEstimator implements the Blake-Zisserman
        M-Estimator.
        \brief Blake-Zisserman M-Estimator
      */
    class BlakeZissermanMEstimator :
      public MEstimator {
    public:
      /** \name Constructors/destructor
        @{
        */
      /// Constructor
      BlakeZissermanMEstimator(size_t df, double pCut = 0.999,
          double wCut = 0.1) :
          _df(df),
          _pCut(pCut),
          _wCut(wCut),
          _epsilon(computeEpsilon(df, pCut, wCut)) {
      };
      /// Copy constructor
      BlakeZissermanMEstimator(const BlakeZissermanMEstimator& other) :
        MEstimator(other),
        _df(other._df),
        _pCut(other._pCut),
        _wCut(other._wCut),
        _epsilon(other._epsilon) {
      };
      /// Assignment operator
      BlakeZissermanMEstimator& operator =
          (const BlakeZissermanMEstimator& other) {
        if (this != &other) {
          MEstimator::operator=(other);
          _df = other._df;
          _pCut = other._pCut;
          _wCut = other._wCut;
          _epsilon = other._epsilon;
        }
        return *this;
      };
      /// Destructor
      virtual ~BlakeZissermanMEstimator() {}
      /** @}
        */

      /** \name Methods
        @{
        */
      /// Evaluate the weight function for a given squared Mahalanobis distance
      virtual double getWeight(double mahalanobis2) {
        return exp(-mahalanobis2) / (exp(-mahalanobis2) + _epsilon);
      }
      /// Returns the ASCII name of the M-Estimator
      virtual std::string name() {
        std::stringstream ss;
        ss << "Blake-Zisserman(" << _epsilon << ")";
        return ss.str();
      }
      /// Returns the inverse chi-squared cdf for p and df
      double chi2InvCDF(double p, size_t df) const {
        // TODO: implement this
        SM_ASSERT_EQ(Exception, p, 0.999, "p should be 0.999");
        SM_ASSERT_GE_LT(Exception, df, 1, 20, "df should be between 1 and 20");
        return chi2invTable[df - 1];
      }
      /// Compute optimal epsilon
      double computeEpsilon(size_t df, double pCut, double wCut) const {
        return (1 - wCut) / wCut * exp(-chi2InvCDF(pCut, df));
      }
      /** @}
        */

      /** \name Members
        @{
        */
      /// Degrees of freedom
      size_t _df;
      /// Probability at which we want to cut
      double _pCut;
      /// Weight to assign at this probability
      double _wCut;
      /// Epsilon
      double _epsilon;
      /** @}
        */

    };

  } // namespace backend
} // namespace aslam


#endif
