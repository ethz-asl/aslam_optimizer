/*!
 * \file ErrorTermEuclidean.hpp
 *
 * \author diemf
 *
 * A Pose ErrorTerm
 */
#ifndef ASLAM_BACKEND_ERROR_EUCLIDEAN_HPP
#define ASLAM_BACKEND_ERROR_EUCLIDEAN_HPP

#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/EuclideanExpression.hpp>
#include <Eigen/Core>

namespace aslam {
  namespace backend {

    /*!
    * \class ErrorTermEuclidean
    *
    * \brief An ErrorTerm implementation for the deviation of a translation pose wrt. a prior
    */

    class ErrorTermEuclidean : public aslam::backend::ErrorTermFs<3>
    {
    public:
      // This is important. The superclass holds some fixed-sized Eigen types
      // For more information, see:
      // http://eigen.tuxfamily.org/dox-devel/TopicStructHavingEigenMembers.html
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      ErrorTermEuclidean(const aslam::backend::EuclideanExpression& t, const Eigen::Vector3d& prior, const Eigen::Matrix<double,3,3>& N);
      ErrorTermEuclidean(const aslam::backend::EuclideanExpression& t, const Eigen::Vector3d& prior, double weight);

      ~ErrorTermEuclidean() override;

    protected:
      /// This is the interface required by ErrorTermFs<>

      /// \brief evaluate the error term and return the weighted squared error e^T invR e
      double evaluateErrorImplementation() override;

      /// \brief evaluate the jacobian
      void evaluateJacobiansImplementation(JacobianContainer & J) override;

    private:
      Eigen::Vector3d _prior;
      aslam::backend::EuclideanExpression _t;
    };

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_ERROR_EUCLIDEAN_HPP */
