#ifndef ASLAM_BACKEND_EUCLIDEAN_DIRECTION_HPP
#define ASLAM_BACKEND_EUCLIDEAN_DIRECTION_HPP

#include "EuclideanExpressionNode.hpp"
#include "EuclideanExpression.hpp"
#include <aslam/backend/DesignVariable.hpp>
#include <sm/kinematics/EulerAnglesYawPitchRoll.hpp>

namespace aslam {
    namespace backend {
    
        class EuclideanDirection : public EuclideanExpressionNode, public DesignVariable
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

        SM_DEFINE_EXCEPTION(Exception, std::runtime_error);

            EuclideanDirection(const Eigen::Vector3d & direction);
            ~EuclideanDirection() override;

            /// \brief Revert the last state update.
            void revertUpdateImplementation() override;

            /// \brief Update the design variable.
            void updateImplementation(const double * dp, int size) override;

            /// \brief the size of an update step
            int minimalDimensionsImplementation() const override;

            vector_t toEuclidean() { return evaluate(); }
            EuclideanExpression toExpression();
        private:
            Eigen::Vector3d evaluateImplementation() const override;

            void evaluateJacobiansImplementation(JacobianContainer & outJacobians) const override;

            void getDesignVariablesImplementation(DesignVariable::set_t & designVariables) const override;

			/// Returns the content of the design variable
			void getParametersImplementation(Eigen::MatrixXd& value) const override;

			/// Sets the content of the design variable
			void setParametersImplementation(const Eigen::MatrixXd& value) override;

		    /// Computes the minimal distance in tangent space between the current value of the DV and xHat
		    void minimalDifferenceImplementation(const Eigen::MatrixXd& xHat, Eigen::VectorXd& outDifference) const override;

		    /// Computes the minimal distance in tangent space between the current value of the DV and xHat and the jacobian
		    void minimalDifferenceAndJacobianImplementation(const Eigen::MatrixXd& xHat, Eigen::VectorXd& outDifference, Eigen::MatrixXd& outJacobian) const override;

            Eigen::Matrix3d _C;
            Eigen::Matrix3d _p_C;

            double _magnitude;

            sm::kinematics::EulerAnglesYawPitchRoll _ypr;

        };
        
    } // namespace backend    
} // namespace aslam


#endif /* ASLAM_BACKEND_UNIT_EUCLIDEAN_HPP */
