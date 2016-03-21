/*
 * MatrixStack.hpp
 *
 *  Created on: 21.03.2016
 *      Author: Ulrich Schwesinger
 */

#ifndef INCLUDE_ASLAM_BACKEND_MATRIXSTACK_HPP_
#define INCLUDE_ASLAM_BACKEND_MATRIXSTACK_HPP_

// standard includes
#include <vector>
#include <utility> // std::move

// Eigen includes
#include <Eigen/Dense>

// Schweizer-Messer
#include <sm/logging.hpp>

// self includes
#include <aslam/Exceptions.hpp>

namespace aslam {
namespace backend {

  /**
   * \class MatrixStack
   * \brief Stores a chain rule of matrices in a stack-like data structure
   */
  class MatrixStack
  {
   public:
    typedef double Scalar;
    typedef Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> Map;
    typedef Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned> ConstMap;

    /**
     * \class Header
     * \brief Metadata for stack elements
     */
    struct Header
    {
      Header(const int rows_, const int cols_, const std::size_t dataIndex_)
          : rows(rows_), cols(cols_), dataIndex(dataIndex_)
      {

      }
      int rows; /// \brief Number of rows of the data
      int cols; /// \brief Number of columns of the data
      std::size_t dataIndex;  /// \brief Index to the beginning of the data of this element
    };

    /**
     * \class PopGuard
     * \brief Ensures that \p pop() is called on the stack when the guard goes
     *        out of scope
     */
    struct PopGuard
    {
      PopGuard(MatrixStack& stack)
          : _stack(&stack)
      {

      }

      PopGuard(const PopGuard&) = delete;
      void operator=(const PopGuard&) = delete;

      PopGuard(PopGuard&& pg)
      {
        *this = std::move(pg);
      }
      void operator=(PopGuard&& pg)
      {
        this->_stack = pg._stack;
        pg._stack = nullptr;
      }

      void pop()
      {
        if (_stack != nullptr) {
          _stack->pop();
          _stack = nullptr;
        }
      }
      ~PopGuard() { this->pop(); }

      MatrixStack& stack()
      {
        SM_ASSERT_NOTNULL(Exception, _stack, ""); // TODO: make debug
        return *_stack;
      }

     private:
      MatrixStack* _stack;
    };

   public:

    /// \brief Constructs a stack
    MatrixStack(const std::size_t maxNumMatrices, const std::size_t estimatedNumElementsPerMatrix)
    {
      _headers.reserve(maxNumMatrices);
      _data.reserve(maxNumMatrices*estimatedNumElementsPerMatrix);
    }

    /// \brief Is the stack empty?
    bool empty() const { return _headers.empty(); }

    /// \brief Number of matrices stored
    std::size_t numMatrices() const { return _headers.size(); }

    /// \brief Number of matrix elements stored
    std::size_t numElements() const { return _data.size(); }

    /// \brief Push a matrix \p mat to the top of the stack
    template <typename DERIVED>
    void push(const Eigen::MatrixBase<DERIVED>& mat)
    {
      if (!this->empty())
      {
        SM_ASSERT_EQ(Exception, this->numTopCols(), mat.rows(), "Incompatible matrix sizes");
        this->allocate(this->numRows(), mat.cols()); // We allocate space for 1 more matrix. Stack wasn't empty before, so now we have at least 2.
        const auto last = this->matrix(this->numMatrices()-2);
        this->top() = last*mat;
      }
      else
      {
        this->allocate(mat.rows(), mat.cols());
        this->top() = mat;
      }
    }

    /// \brief Push a matrix \p mat to the top of the stack
    template <typename DERIVED>
    PopGuard pushWithGuard(const Eigen::MatrixBase<DERIVED>& mat)
    {
      this->push(mat);
      return PopGuard(*this);
    }

    /// \brief Pop the top element from the stack
    void pop()
    {
      SM_ASSERT_FALSE(Exception, this->empty(), "pop() on an empty stack is forbidden!");
      const int newsz = this->numElements() - _headers.back().rows*_headers.back().cols;
      SM_ASSERT_GE_DBG(Exception, newsz, 0, "Something went wrong, this is a bug in the code!");
      _data.resize(newsz);
      _headers.pop_back();
    }

    /// \brief Const getter for the top matrix in the stack
    ConstMap top() const {
      SM_ASSERT_FALSE(Exception, this->empty(), "");
      return ConstMap( &(_data[_headers.back().dataIndex]), _headers.back().rows, _headers.back().cols );
    }

    /// \brief Mutable getter for the top matrix in the stack
    Map top() {
      SM_ASSERT_FALSE(Exception, this->empty(), "");
      return Map( &(_data[_headers.back().dataIndex]), _headers.back().rows, _headers.back().cols );
    }

   private:

    /// \brief Const getter for the \p i-th matrix in the stack
    ConstMap matrix(const std::size_t i) const {
      SM_ASSERT_LT(IndexOutOfBoundsException, i, this->numMatrices(), "");
      return ConstMap( &(_data[_headers[i].dataIndex]), _headers[i].rows, _headers[i].cols );
    }

    /// \brief Allocates memory and metadata for an element of size \p rows x \p cols.
    void allocate(const int rows, const int cols)
    {
      static constexpr int align = 16/sizeof(Scalar);
      std::size_t start = this->numElements();
      if (start % align != 0) // manual memory alignment
        start += align - start % align;
#ifndef NDEBUG // warn in debug compilation mode if memory has to be reallocated
      SM_WARN_STREAM_COND_NAMED(start + rows*cols > _data.capacity(), "optimization", "Matrix stack has to reallocate memory. "
          "Consider constructing the matrix stack with more memory reserves for faster execution.");
#endif
      _data.resize(start + rows*cols); // Note: If memory has to be allocated because new size is bigger than capacity, all iterators will be invalidated
      _headers.emplace_back(rows, cols, start);
    }

    /// \brief Number of columns of the matrix on top of the stack
    int numTopCols() const {
      return _headers.back().cols;
    }

    /// \brief Number of rows determined by the first matrix that was pushed
    int numRows() const {
      return _headers.front().rows;
    }

   private:
    typedef Eigen::aligned_allocator< std::vector<Scalar> > AlignedAllocator;
    std::vector<Scalar, AlignedAllocator> _data; /// \brief The data of the matrices
    std::vector<Header> _headers; /// \brief Metadata for the matrix entries
  };

} /* namespace aslam */
} /* namespace backend */

#endif /* INCLUDE_ASLAM_BACKEND_MATRIXSTACK_HPP_ */
