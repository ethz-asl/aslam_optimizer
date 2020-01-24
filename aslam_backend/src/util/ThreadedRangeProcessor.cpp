#include <aslam/backend/util/ThreadedRangeProcessor.hpp>

#include <exception>
#include <future>

#include <boost/thread.hpp>

#include <sm/logging.hpp>
#include <sm/assert_macros.hpp>

namespace aslam {
namespace backend {
namespace util {

/// \brief The return value for a safe job
struct SafeJobReturnValue {
  SafeJobReturnValue(const std::exception& e) : _e(e) {}
  std::exception _e;
};

/// \brief Functor running a job catching all exceptions
struct SafeJob {
  boost::function<void()> _fn;
  SafeJobReturnValue* _rval;
  SafeJob() : _rval(NULL) {}
  SafeJob(boost::function<void()> fn) : _fn(fn), _rval(NULL) {}
  ~SafeJob() {
    if (_rval) delete _rval;
  }

  void operator()() {
    try {
      _fn();
    } catch (const std::exception& e) {
      _rval = new SafeJobReturnValue(e);
      SM_FATAL_STREAM("Exception in thread block: " << e.what());
    }
  }
};

void runThreadedJob(boost::function<void(size_t, size_t, size_t)> job, size_t rangeLength, size_t nThreads)
{
  SM_ASSERT_GT(std::runtime_error, nThreads, 0, "");
  if (rangeLength == 0) // nothing to process here
    return;

  if (nThreads == 1) {
    job(0, 0, rangeLength);
  } else {
    nThreads = std::min(nThreads, rangeLength);

    // Compute the sub-ranges for each thread.
    std::vector<int> indices(nThreads + 1, 0);
    int nJPerThread = std::max(1, static_cast<int>(rangeLength / nThreads));
    for (unsigned i = 0; i < nThreads; ++i)
      indices[i + 1] = indices[i] + nJPerThread;
    // deal with the remainder.
    indices.back() = rangeLength;

    // replace by std::async for performance
    std::vector<std::future<void>> jobs;
    jobs.reserve(nThreads);
    for (unsigned i = 0; i < nThreads; ++i) {
      jobs.push_back(
          std::async(std::launch::async, [job, i, &indices]() {
            job(i, indices[i], indices[i + 1]);
          }));
    }
    for (auto& j : jobs) {
      j.get();
    }
  }
}

}
}
}

