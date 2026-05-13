#pragma once
// ============================================================
// thread_pool.hpp — spinwait fork-join thread pool
//
// Design goal: minimal barrier latency for frequent short tasks.
//
// Key properties:
//   - Workers spin on an atomic generation counter (no futex syscalls)
//   - Calling thread participates in computation (1 extra chunk)
//   - Barrier overhead: ~200ns vs 10-100μs for mutex+condvar
//   - Workers burn CPU during idle (ideal for continuous inference)
//
// Usage:
//   ThreadPool pool(n_workers);   // n_workers parallel threads
//   pool.parallel_for(N, [&](int s, int e) {
//       for (int i = s; i < e; i++) process(i);
//   });
//   // Calling thread also does 1/(n_workers+1) of the work
// ============================================================
#include <atomic>
#include <functional>
#include <immintrin.h>   // _mm_pause
#include <thread>
#include <vector>
#ifdef __linux__
#  include <pthread.h>
#  include <sched.h>
#endif

namespace axonforge {

class ThreadPool {
public:
    explicit ThreadPool(int n_workers) {
        workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; i++)
            workers_.emplace_back([this, i] { pin_to_cpu_(i); worker_loop(i); });
        // Wait until all workers are spinning before returning
        while (live_count_.load(std::memory_order_acquire) < n_workers)
            _mm_pause();
    }

    // Pin the calling thread to a specific logical CPU (call from llama_generate
    // after pool construction to give the serial calling thread its own CPU).
    static void pin_caller(int cpu_id) noexcept { pin_to_cpu_(cpu_id); }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        for (auto& t : workers_) t.join();
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const noexcept { return static_cast<int>(workers_.size()); }

    // Partition [0, N) into (n_workers+1) chunks.
    // Workers handle chunks 0..n_workers-1; calling thread handles the last chunk.
    // Blocks until all workers finish.  fn must be thread-safe.
    void parallel_for(int N, std::function<void(int, int)> fn) {
        const int nw = static_cast<int>(workers_.size());
        if (nw == 0) { fn(0, N); return; }

        const int n     = nw + 1;          // workers + calling thread
        const int chunk = (N + n - 1) / n;

        // Publish task — release order ensures workers see task_fn_/task_N_/task_chunk_
        task_fn_    = fn;                  // write before generation bump
        task_N_     = N;
        task_chunk_ = chunk;
        pending_.store(nw, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);

        // Calling thread does the last chunk concurrently with workers
        const int cs = nw * chunk;
        const int ce = std::min(N, cs + chunk);
        if (cs < ce) fn(cs, ce);

        // Spinwait for workers to complete
        while (pending_.load(std::memory_order_acquire) > 0)
            _mm_pause();
    }

private:
    std::vector<std::thread>      workers_;
    std::atomic<int>              generation_{0};
    std::atomic<int>              pending_{0};
    std::atomic<bool>             stop_{false};
    std::atomic<int>              live_count_{0};

    // Task descriptor — written before generation bump, read after acquire load
    std::function<void(int, int)> task_fn_;
    int  task_N_     = 0;
    int  task_chunk_ = 0;

    static void pin_to_cpu_(int cpu_id) noexcept {
#ifdef __linux__
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu_id, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
    }

    void worker_loop(int id) {
        live_count_.fetch_add(1, std::memory_order_release);
        int last_gen = 0;
        for (;;) {
            // Bounded spin (~0.5ms @ 10ns/pause), then yield to avoid OS starvation
            static constexpr int kSpinBudget = 50000;
            int g, spins = 0;
            while ((g = generation_.load(std::memory_order_acquire)) == last_gen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
                if (++spins >= kSpinBudget) {
                    spins = 0;
                    std::this_thread::yield();
                }
            }
            if (stop_.load(std::memory_order_relaxed)) return;
            last_gen = g;

            const int s = id * task_chunk_;
            const int e = std::min(s + task_chunk_, task_N_);
            if (s < e) task_fn_(s, e);

            pending_.fetch_sub(1, std::memory_order_release);
        }
    }
};

} // namespace axonforge
