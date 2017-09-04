#include <aslam/backend/ToTextNodeVisitor.h>

namespace aslam {
namespace backend {

ToTextNodeVisitor::ToTextNodeVisitor() = default;
ToTextNodeVisitor::~ToTextNodeVisitor() = default;

void ToTextNodeVisitor::nullNodeV() {
  text << "<>";
}

void ToTextNodeVisitor::visitString(const char * c) {
  text << c;
}

void ToTextNodeVisitor::visitV(NodeI & node) {
  text << node.getName();
  if (node.hasArgs()){
    text << "(";
    for(size_t i = 0; i < node.getNumArgs(); i++){
      node.accept(i, *this);
      if (i+1 < node.getNumArgs()) text << ", ";
    }
    text << ")";
  }
  text << "=" << node.computeValue();
}

} /* namespace backend */
} /* namespace aslam */
