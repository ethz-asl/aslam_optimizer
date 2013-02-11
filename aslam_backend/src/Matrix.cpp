#include <aslam/backend/Matrix.hpp>

namespace aslam {
  namespace backend {

    Matrix::Matrix()
    {
    }

    Matrix::~Matrix()
    {
    }

    Eigen::MatrixXd Matrix::toDense() const
    {
      Eigen::MatrixXd M;
      toDenseInto(M);
      return M;
    }

    void Matrix::toDenseInto(Eigen::MatrixXd& outM) const
    {
      outM.resize(rows(), cols());
      for (size_t c = 0; c < cols(); ++c) {
        for (size_t r = 0; r < rows(); ++r) {
          outM(r, c) = (*this)(r, c);
        }
      }
    }



  } // namespace backend
} // namespace aslam


std::ostream& operator<<(std::ostream& os, const aslam::backend::Matrix& ccm)
{
  for (size_t r = 0; r < ccm.rows(); ++r) {
    for (size_t c = 0; c < ccm.cols(); ++c) {
      os.width(10);
      os.setf(std::ios::fixed, std::ios::floatfield);  // floatfield set to fixed
      os.precision(5);
      os << ccm(r, c) << " ";
    }
    os << std::endl;
  }
  return os;
}
