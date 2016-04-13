/*
 * CostFunctionInterface.hpp
 *
 *  Created on: 13.04.2016
 *      Author: Ulrich Schwesinger (ulrich.schwesinger@mavt.ethz.ch)
 */

#ifndef INCLUDE_ASLAM_BACKEND_UTIL_COSTFUNCTIONINTERFACE_HPP_
#define INCLUDE_ASLAM_BACKEND_UTIL_COSTFUNCTIONINTERFACE_HPP_

#include <aslam/backend/util/CommonDefinitions.hpp> //RowVectorType, ColumnVectorType


namespace aslam
{
namespace backend
{

/**
 * \struct CostFunctionInterface
 * Generic interface for any cost function
 */
struct CostFunctionInterface
{
  virtual ~CostFunctionInterface() { }
  virtual double evaluateError() const = 0;
  virtual void computeGradient(RowVectorType& gradient) = 0;
  virtual void applyStateUpdate(const ColumnVectorType& dx) = 0;
  virtual Eigen::VectorXd getFlattenedDesignVariableParameters() = 0;
  virtual void saveDesignVariables() = 0;
  virtual void restoreDesignVariables() = 0;
};

} /* namespace aslam */
} /* namespace backend */


#endif /* INCLUDE_ASLAM_BACKEND_UTIL_COSTFUNCTIONINTERFACE_HPP_ */
