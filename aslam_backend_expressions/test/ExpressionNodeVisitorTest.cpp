#include <sm/eigen/gtest.hpp>
#include <aslam/backend/ScalarExpression.hpp>
#include <aslam/backend/EuclideanExpression.hpp>
#include <aslam/backend/ExpressionNodeVisitor.hpp>
#include <aslam/backend/ToTextNodeVisitor.h>

using namespace aslam::backend;


class TestNode {
 public:
  TestNode(int v) : v_(v) {}

  void accept(ExpressionNodeVisitor & v){
    v.visit("N", this);
  }

  int evaluate() {
    return v_;
  }
  int v_;
};

class TextNode {
 public:
  void accept(ExpressionNodeVisitor & v){
    v.visit("T");
  }
};

class TestNodeWithArgs {
 public:
  void accept(ExpressionNodeVisitor & v){
    v.visit("NWA", this, & a, & b);
  }

  int evaluate() {
    return a.v_ + b.v_;
  }

  TestNode a = 3, b = 5;
};

TEST(NodeVisitorSuite, testSingle)
{
  ToTextNodeVisitor v;

  TestNode n(5);
  n.accept(v);

  EXPECT_EQ("N=5", v.getText());
}

TEST(NodeVisitorSuite, testEmpty)
{
  ToTextNodeVisitor v;

  boost::shared_ptr<TestNode> ptr(nullptr);
  v.beAcceptedBy(ptr);

  EXPECT_EQ("<>", v.getText());
}


TEST(NodeVisitorSuite, testWithArgs)
{
  ToTextNodeVisitor v;

  TestNodeWithArgs nwa;
  nwa.accept(v);

  EXPECT_EQ("NWA(N=3, N=5)=8", v.getText());
}


TEST(NodeVisitorSuite, testScalarExp)
{
  ToTextNodeVisitor v;

  ScalarExpression a(3), b(5);
  auto c = a + b;

  c.accept(v);

  EXPECT_EQ("+(#=3, #=5)=8", v.getText());
}

TEST(NodeVisitorSuite, testScalarExpNamed)
{
  ToTextNodeVisitor v;

  ScalarExpression a("a", 3), b("b", 5);
  auto c = a + b;

  c.accept(v);

  EXPECT_EQ("+(a=3, b=5)=8", v.getText());
}

TEST(NodeVisitorSuite, testEuclideanExp)
{
  ToTextNodeVisitor v;

  EuclideanExpression a({3, 5 , 6}), b({1, 1, 1});
  EuclideanExpression c = a + b;

  c.accept(v);

  EXPECT_EQ("+(#=3\n5\n6, #=1\n1\n1)=4\n6\n7", v.getText());
}

