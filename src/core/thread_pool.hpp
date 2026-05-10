#pragma once
// ============================================================
// thread_pool.hpp — lightweight fork-join thread pool
//
// Design goal: zero per-call allocation, minimal lock contention.
//
// Usage:
//   ThreadPool pool(8);
//   pool.parallel_for(N, [&](int s, int e) {
//       for (int i = s; i < e; i++) process(i);
//   });
//
// parallel_for() is blocking: returns only when all workers finish.
// The pool is reused across calls — workers sleep between calls.
// ============================================================
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace axonforge {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) {
        workers_.reserve(n_threads);
        for (int i = 0; i < n_threads; i++)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Not copyable/movable (threads hold a pointer to this)
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const noexcept { return static_cast<int>(workers_.size()); }

    // Partition [0, N) into chunks and run fn(chunk_start, chunk_end) per worker.
    // fn must be thread-safe.  Blocks until all workers finish.
    void parallel_for(int N, std::function<void(int, int)> fn) {
        if (workers_.empty()) { fn(0, N); return; }

        const int n  = static_cast<int>(workers_.size());
        const int chunk = (N + n - 1) / n;

        {
            std::unique_lock<std::mutex> lk(mtx_);
            task_fn_     = std::move(fn);
            task_N_      = N;
            task_chunk_  = chunk;
            pending_     = n;
            generation_++;
        }
        cv_work_.notify_all();

        // Wait for all workers to finish
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    std::vector<std::thread>       workers_;
    std::mutex                     mtx_;
    std::condition_variable        cv_work_;
    std::condition_variable        cv_done_;

    // Task state (protected by mtx_)
    std::function<void(int, int)>  task_fn_;
    int  task_N_     = 0;
    int  task_chunk_ = 0;
    int  pending_    = 0;
    int  generation_ = 0;   // incremented per parallel_for() to distinguish tasks
    bool stop_       = false;

    void worker_loop(int id) {
        int last_gen = 0;
        for (;;) {
            // Wait for a new task (generation changes) or stop signal
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [&] { return generation_ != last_gen || stop_; });
            if (stop_) return;

            last_gen = generation_;
            const int s   = id * task_chunk_;
            const int e   = std::min(s + task_chunk_, task_N_);
            auto fn       = task_fn_;   // copy shared_ptr counter (cheap)
            lk.unlock();

            if (s < e) fn(s, e);

            lk.lock();
            if (--pending_ == 0) cv_done_.notify_one();
        }
    }
};

} // namespace axonforge
