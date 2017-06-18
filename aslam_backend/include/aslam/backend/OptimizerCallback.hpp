#ifndef OPTIMIZER_CALLBACK_HPP
#define OPTIMIZER_CALLBACK_HPP

#include <boost/make_shared.hpp>

namespace aslam {
namespace backend {
namespace callback {

/**
 * \brief The callback event specifies the callback injection point in the optimizer.
 */

struct Event {
  Event(
      double currentCost_ = std::numeric_limits<double>::signaling_NaN(),
      double previousLowestCost_ = std::numeric_limits<double>::signaling_NaN()
      )
      : currentCost(currentCost_), previousLowestCost(previousLowestCost_)
  {
  }
  virtual ~Event() = default;

  /**
   * \brief the most recently evaluated effective cost (J = sum of squared errors after the m-estimators being applied).
   * Its relation to the optimization phase depends on the callback event:
   *   OPTIMIZATION_INITIALIZED:      the initial value (before any optimization)
   *   DESIGN_VARIABLES_UPDATED, RESIDUALS_UPDATED:  undefined
   *   COST_UPDATED:                  the new cost after an update of the design variables
   */
  double currentCost;
  /**
   * \brief the most recently evaluated effective cost (J = sum of squared errors after the m-estimators being applied).
   * Its relation to the optimization phase depends on the callback event:
   *   OPTIMIZATION_INITIALIZED, DESIGN_VARIABLES_UPDATED, RESIDUALS_UPDATED:  undefined
   *   COST_UPDATED:                  the lowest previous cost so far (or -1 if this is the initial update)
   */
  double previousLowestCost;
};

namespace event {

/// \brief Right after the the initial cost has been computed.
struct OPTIMIZATION_INITIALIZED : Event {
  using Event::Event;
};

/// \brief At the start of an iteration before any work has been done, same state as in PER_ITERATION_END in previous iteration
struct ITERATION_START : Event {
  using Event::Event;
};

/// \brief At the end of an iteration after all work has been done, same state as in PER_ITERATION_START in next iteration
struct ITERATION_END : Event {
  using Event::Event;
};

/// \brief Right after the linear system was solved
struct LINEAR_SYSTEM_SOLVED : Event {
  using Event::Event;
};

/// \brief Right after the design variables (X) have been updated.
struct DESIGN_VARIABLES_UPDATED : Event {
  using Event::Event;
};

/// \brief After the raw squared error for each error term has been updated but before the m-estimators are applied to compute the effective cost. This is the right event to update m-estimators based on residuals.
struct RESIDUALS_UPDATED : Event {
  using Event::Event;
};

/// \brief Right after the m-estimators are applied to compute the effective cost.
struct COST_UPDATED : Event {
  using Event::Event;
};

}

enum class ProceedInstruction {
  CONTINUE,
  SUCCEED,
  FAIL
};

class OptimizerCallbackInterface {
public:
  typedef boost::shared_ptr<OptimizerCallbackInterface> Ptr;
  virtual ~OptimizerCallbackInterface() {}
  virtual ProceedInstruction operator() (const Event & arg) = 0;
};

template <typename Funct, typename Event_>
class CallbackFunctor : public OptimizerCallbackInterface {
public:
  CallbackFunctor(Funct f) : f(f) {}
  ProceedInstruction operator() (const Event & arg) override {
    return call(arg);
  }
private:
  template<typename F = Funct, std::is_same<ProceedInstruction, decltype((*static_cast<F*>(nullptr))(*static_cast<Event_*>(nullptr)))>* returnsProceedInstruction = nullptr>
  ProceedInstruction call(const Event & arg) {
    return withArg(arg, returnsProceedInstruction);
  }
  template<typename F = Funct, std::is_same<ProceedInstruction, decltype((*static_cast<F*>(nullptr))())>* returnsProceedInstruction = nullptr>
  ProceedInstruction call(const Event &) {
    return withoutArg(returnsProceedInstruction);
  }

  ProceedInstruction withoutArg(std::true_type*) {
    return f();
  }
  ProceedInstruction withoutArg(std::false_type*) {
    f();
    return ProceedInstruction::CONTINUE;
  }
  ProceedInstruction withArg(const Event & arg, std::true_type*) {
    return f(static_cast<const Event_ &>(arg));
  }
  ProceedInstruction withArg(const Event & arg, std::false_type*) {
    f(static_cast<const Event_ &>(arg));
    return ProceedInstruction::CONTINUE;
  }
  Funct f;
};

template <typename Event_ = Event, typename Funct>
OptimizerCallbackInterface::Ptr toOptimizerCallback(Funct f){
  return boost::make_shared<CallbackFunctor<Funct, Event_>>(f);
}

class OptimizerCallback {
public:
  OptimizerCallback(const OptimizerCallbackInterface::Ptr & impl) : impl_(impl) {}

  template <typename Funct>
  OptimizerCallback(Funct funct) : OptimizerCallback(toOptimizerCallback(funct)) {}

  template <typename Event_, typename Funct>
  OptimizerCallback(Event_*, Funct funct) : OptimizerCallback(toOptimizerCallback<Event_>(funct)) {}

  bool operator == (const OptimizerCallback & other) const { return impl_ == other.impl_; };
  ProceedInstruction operator() (const Event & arg){
    return (*impl_)(arg);
  }
private:
  OptimizerCallbackInterface::Ptr impl_;
};



} // namespace callback
} // namespace backend
}  // namespace aslam

#endif /* OPTIMIZER_CALLBACK_HPP */
