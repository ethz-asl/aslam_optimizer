#ifndef H6447EE20_3FB1_40F3_8DFB_E566F2F93842
#define H6447EE20_3FB1_40F3_8DFB_E566F2F93842
#include <memory>
#include <string>
#include <sstream>
#include <typeinfo>
#include <tuple>

#include <boost/any.hpp>

namespace aslam {
namespace backend {


class ExpressionNodeVisitor;

class NodeI {
 public:
  virtual ~NodeI() = default;
  virtual const char * getName() const = 0;
  virtual std::string computeValue() const = 0;
  virtual void accept(size_t argIndex, ExpressionNodeVisitor &) const = 0;
  virtual size_t getNumArgs() const = 0;

  bool hasArgs() const { return getNumArgs(); }
};


class ExpressionNodeVisitor {
 public:
  virtual ~ExpressionNodeVisitor();

  void visit(const char *c){
    visitString(c);
  }
  template <typename T>
  void visitAny(T *t){
    visitTypeInfo(typeid(*t), t);
  }

  template <typename ExpNode, typename ... Args>
  void visit(const char *, const ExpNode & node, Args ... args);

  template <typename ExpNode, typename ... Args>
  void visit(const char *, const ExpNode & node, const std::tuple<Args ...>& args);

  template <typename ExpNode, typename ... Args>
  void visit(const ExpNode & node, Args ... args) {
    visit(typeid(node).name(), node, args...);
  }


  template <typename Arg>
  void beAcceptedBy(Arg * ptr) {
    if(ptr == nullptr){
      nullNodeV();
    } else {
      ptr->accept(*this);
    }
  }

  template <typename Arg>
  void beAcceptedBy(const Arg & ref, int = 0.) {
    ref.accept(*this);
  }

  template <typename Arg>
  void beAcceptedBy(const boost::shared_ptr<Arg> & ptr) {
    beAcceptedBy(ptr.get());
  }

  template <typename Arg>
  void beAcceptedBy(const std::shared_ptr<Arg> & ptr) {
    beAcceptedBy(ptr.get());
  }
 private:
  virtual void visitV(NodeI & node) = 0;
  virtual void visitString(const char * text) = 0;
  virtual void visitTypeInfo(const std::type_info & type_info, void * /*ptr*/) {
    visitString(type_info.name());
  }
  virtual void nullNodeV() = 0;
};

template <typename ExpNode, typename ... Args>
class NodeImpl : public NodeI {
 public:
  NodeImpl(const char * name, const ExpNode & n, Args ... args) : name_(name), n(n), args_(args...) {}
  NodeImpl(const char * name, const ExpNode & n, const std::tuple<Args...>& args) : name_(name), n(n), args_(args) {}
  virtual ~NodeImpl() = default;
  virtual const char * getName() const override {
    return name_;
  }
  virtual std::string computeValue() const override {
    std::stringstream s;
    s << n->evaluate();
    return s.str();
  }

  virtual void accept(size_t argIndex, ExpressionNodeVisitor & v) const override {
    if(argIndex >= numArgs()){
      throw std::runtime_error("");
    }
    visitArg<static_cast<int>(numArgs()) - 1>(argIndex, v);
  }

  virtual size_t getNumArgs() const override {
    return numArgs();
  }
 private:

  template <int I>
  typename std::enable_if<(I >= 0)>::type visitArg(size_t argIndex, ExpressionNodeVisitor &v) const {
    if(I == argIndex){
      v.beAcceptedBy(std::get<I>(args_));
    } else {
      visitArg<I-1>(argIndex, v);
    }
  }

  template <int I>
  typename std::enable_if<(I < 0)>::type visitArg(size_t, ExpressionNodeVisitor &) const {
  }

  static constexpr size_t numArgs() {
    return std::tuple_size<decltype(args_)>::value;
  }

  const char * name_;
  const ExpNode & n;
  std::tuple<Args...> args_;
};



template <typename ExpNode, typename ... Args>
void ExpressionNodeVisitor::visit(const char * name, const ExpNode & node, Args ... args) {
  NodeImpl<ExpNode, Args...> nodeImpl(name, node, args...);
  visitV(nodeImpl);
}

template <typename ExpNode, typename ... Args>
void ExpressionNodeVisitor::visit(const char * name, const ExpNode & node, const std::tuple<Args ...>& args) {
  NodeImpl<ExpNode, Args...> nodeImpl(name, node, args);
  visitV(nodeImpl);
}



} // namespace backend
    }  // namespace aslam

#endif /* H6447EE20_3FB1_40F3_8DFB_E566F2F93842 */
