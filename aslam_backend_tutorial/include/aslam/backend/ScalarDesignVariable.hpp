#ifndef ASLAM_BACKEND_SCALAR_DESIGN_VARIABLE
#define ASLAM_BACKEND_SCALAR_DESIGN_VARIABLE

#include <aslam/backend/DesignVariable.hpp>
namespace aslam {
  namespace backend {
    
    class ScalarDesignVariable : public DesignVariable
    {
    public:
      /// \brief initialize the variable with some initial guess
      ScalarDesignVariable(double initialValue);
      virtual ~ScalarDesignVariable();
      
      /// \brief get the current value. This is not required but it will be used
      // by error terms to get the current value of the design variable.
      double value();
    protected:
      // The following three virtual functions are required by DesignVariable

      /// \brief what is the number of dimensions of the perturbation variable.
      virtual int minimalDimensionsImplementation() const;
      
      /// \brief Update the design variable.
      virtual void updateImplementation(const double * dp, int size);
      
      /// \brief Revert the last state update.
      virtual void revertUpdateImplementation();
      
    private:
      /// \brief the value of the design variable
      double _value;
      /// \brief the previous version of the design variable
      double _p_value;
    };

  } // namespace backend
} // namespace aslam

#endif /* ASLAM_BACKEND_SCALAR_DESIGN_VARIABLE */
