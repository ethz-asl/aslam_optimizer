#ifndef ASLAM_MESTIMATOR_POLICIES_HPP
#define ASLAM_MESTIMATOR_POLICIES_HPP

#include <cmath>
#include <string>
#include <sstream>

namespace aslam {
  namespace backend {

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

  } // namespace backend
} // namespace aslam


#endif
