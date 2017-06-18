#ifndef ASLAM_MESTIMATOR_POLICIES_HPP
#define ASLAM_MESTIMATOR_POLICIES_HPP

#include <string>
#include <boost/shared_ptr.hpp>

namespace aslam {
  namespace backend {

    class MEstimator {
    public:
      typedef boost::shared_ptr<MEstimator> Ptr;
      virtual ~MEstimator();
      virtual double getWeight(double squaredError) const = 0;
      virtual std::string name() const = 0;
    };

    class NoMEstimator : public MEstimator {
    public:
      ~NoMEstimator() override;
      double getWeight(double squaredError) const override;
      std::string name() const override;
    };

    class  GemanMcClureMEstimator : public MEstimator {
    public:
      GemanMcClureMEstimator(double sigma2);
      ~GemanMcClureMEstimator() override;
      double getWeight(double error) const override;
      std::string name() const override;

      double _sigma2;
    };

    class  CauchyMEstimator : public MEstimator {
    public:
      CauchyMEstimator(double sigma2);
      ~CauchyMEstimator() override;
      double getWeight(double error) const override;
      std::string name() const override;

      double _sigma2;
    };


  /// \class FixedWeightMEstimator
  /// weights errors by a fixed, user supplied weight
    class FixedWeightMEstimator : public MEstimator {
    public:
      FixedWeightMEstimator(double weight);
      ~FixedWeightMEstimator() override;
      double getWeight(double error) const override;
      virtual void setWeight(double weight);
      std::string name() const override;

      double _weight;
    };

  
    class HuberMEstimator : public MEstimator {
    public:
      HuberMEstimator(double k);
      ~HuberMEstimator() override;
      double getWeight(double error) const override;
      std::string name() const override;

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

        /** 
         * The constructor solves for the free parameter in the M-Estimator
         * based on your desired behavior (statistically speaking)
         * 
         * @param df    The dimension of your error term
         * @param pCut  The probability of error that you want to down-weight. 0.99 means 
         *              that the 99% most likely squared errors will not be much down-weighted
         *              and the remaining 1% will be smoothly cut to zero weight.
         * @param wCut  The weight (between 0 and 1 not inclusive) that you want the above probability to have.
         * 
         * @return 
         */
      BlakeZissermanMEstimator(size_t df, double pCut = 0.999,
          double wCut = 0.1);
      /// Copy constructor
      BlakeZissermanMEstimator(const BlakeZissermanMEstimator& other);
      /// Assignment operator
      BlakeZissermanMEstimator& operator =
          (const BlakeZissermanMEstimator& other);
      /// Destructor
      ~BlakeZissermanMEstimator() override;
      /** @}
        */

      /** \name Methods
        @{
        */
      /// Evaluate the weight function for a given squared Mahalanobis distance
      double getWeight(double mahalanobis2) const override;
      /// Returns the ASCII name of the M-Estimator
      std::string name() const override;
      /// Returns the inverse chi-squared cdf for p and df
      double chi2InvCDF(double p, size_t df) const;
      /// Compute optimal epsilon
      double computeEpsilon(size_t df, double pCut, double wCut) const;
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
