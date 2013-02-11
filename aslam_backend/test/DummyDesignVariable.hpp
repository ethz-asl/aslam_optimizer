#ifndef _DUMMYDESIGNVARIABLE_H_
#define _DUMMYDESIGNVARIABLE_H_

template<int MD = 3>
class DummyDesignVariable : public aslam::backend::DesignVariable {
public:
  DummyDesignVariable() {}
  virtual ~DummyDesignVariable() {}
protected:
  /// \brief Revert the last state update.
  virtual void revertUpdateImplementation() {}

  /// \brief Update the design variable.
  virtual void updateImplementation(const double* dp, int size) {}

  /// \brief what is the number of dimensions of the perturbation variable.
  virtual int minimalDimensionsImplementation() const {
    return MD;
  }

};



#endif /* _DUMMYDESIGNVARIABLE_H_ */
