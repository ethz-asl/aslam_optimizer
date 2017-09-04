#ifndef H86E70E95_CA7C_45BC_B837_6957ADDACA3A
#define H86E70E95_CA7C_45BC_B837_6957ADDACA3A
#include <aslam/backend/ExpressionNodeVisitor.hpp>

namespace aslam {
namespace backend {

class ToTextNodeVisitor : public ExpressionNodeVisitor {
 public:
  ToTextNodeVisitor();
  ~ToTextNodeVisitor() override;

  std::string getText() { return text.str(); }

 private:
  virtual void nullNodeV() override;
  virtual void visitString(const char * c) override;
  virtual void visitV(NodeI & node) override;

  std::stringstream text;
};


} /* namespace backend */
} /* namespace aslam */

#endif /* H86E70E95_CA7C_45BC_B837_6957ADDACA3A */
