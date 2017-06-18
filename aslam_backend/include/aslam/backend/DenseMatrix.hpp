#ifndef ASLAM_BACKEND_DENSE_MATRIX_HPP
#define ASLAM_BACKEND_DENSE_MATRIX_HPP

#include "Matrix.hpp"

namespace aslam {
  namespace backend {

    /// \class DenseMatrix
    /// \brief A very simple matrix wrapper.
    class DenseMatrix : public Matrix {
    public:
      DenseMatrix();

      ~DenseMatrix() override;

      /// \brief Get a value from the matrix at row r and column c
      double operator()(size_t r, size_t c) const override;

      /// \brief The number of rows in the matrix.
      size_t rows() const override;

      /// \brief The number of columns in the matrix.
      size_t cols() const override;

      /// \brief Fill and return a dense matrix.
      Eigen::MatrixXd toDense() const;

      /// \brief Fill the input dense matrix. The default version just
      ///        Goes through the matrix calling operator(). Please override.
      void toDenseInto(Eigen::MatrixXd& outM) const override;

      /// \brief Initialize the matrix from a dense matrix
      void fromDense(const Eigen::MatrixXd& M) override;

      /// \brief Initialize the matrix from a dense matrix.
      ///        Entries with absolute value less than tolerance
      ///        should be considered zeros.
      void fromDenseTolerance(const Eigen::MatrixXd& M, double tolerance) override;

      /// \brief right multiply the vector y = A x
      void rightMultiply(const Eigen::VectorXd& x, Eigen::VectorXd& outY) const override;

      /// \brief left multiply the vector y = A^T x
      void leftMultiply(const Eigen::VectorXd& x, Eigen::VectorXd& outY) const override;

      Eigen::MatrixXd _M;

    };

  } // namespace backend
} // namespace aslam


#endif /* ASLAM_BACKEND_DENSE_MATRIX_HPP */
