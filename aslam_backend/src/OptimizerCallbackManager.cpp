#include <aslam/backend/OptimizerCallbackManager.hpp>
#include <map>
#include <vector>
#include <algorithm>
#include <typeindex>

namespace aslam {
namespace backend {
namespace callback {

class RegistryData{
 public:
  std::map<std::type_index, std::vector<OptimizerCallback>> callbacks;
};

Registry::Registry() {
  data = new RegistryData();
}

Registry::~Registry() {
  delete data;
}

void Registry::add(std::initializer_list<std::type_index> events, const OptimizerCallback & callback) {
  for(auto event : events){
    data->callbacks[event].push_back(callback);
  }
}
void Registry::remove(std::initializer_list<std::type_index> events, const OptimizerCallback & callback) {
  for(auto event : events){
    std::vector<OptimizerCallback> & vec = data->callbacks[event];
    auto p = std::remove(vec.begin(), vec.end(), callback);
    vec.erase(p, vec.end());
  }
}
void Registry::clear() {
  data->callbacks.clear();
}

void Registry::clear(std::type_index event) {
  data->callbacks[event].clear();
}

std::size_t Registry::numCallbacks(std::type_index event) const {
  return data->callbacks[event].size();
}

ProceedInstruction Manager::issueCallback(const Event & arg) {
  for(auto & c : data->callbacks[std::type_index(typeid(arg))]){
    auto r =  c(arg);
    if(r != ProceedInstruction::CONTINUE){
      return r;
    }
  }
  return ProceedInstruction::CONTINUE;
}

}  // namespace callback
}  // namespace backend
}  // namespace aslam

