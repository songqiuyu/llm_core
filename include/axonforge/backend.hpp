#pragma once
#include "axonforge/tensor.hpp"
#include "axonforge/graph.hpp"
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace axonforge {

// ============================================================
// IKernel — a single compiled op kernel on a specific backend.
// Backends create and cache IKernel instances; the Scheduler
// holds pointers to them and calls execute() per forward pass.
// ============================================================
class IKernel {
public:
    virtual ~IKernel() = default;

    // Execute the kernel.
    // inputs/outputs are non-owning views: caller manages lifetime.
    virtual void execute(std::span<const Tensor> inputs,
                         std::span<Tensor>       outputs,
                         const AttributeMap&     attrs) = 0;
};

// ============================================================
// IBackend — plugin interface for hardware backends.
//
// Design rules:
//   1. Pure virtual — no default implementations that touch hardware.
//   2. Lifecycle: initialize() → [use] → finalize()
//   3. Kernel dispatch: can_handle() → get_kernel() → IKernel::execute()
//   4. New backends only implement this interface + call
//      AXONFORGE_REGISTER_BACKEND once. No other files change.
// ============================================================
class IBackend {
public:
    virtual ~IBackend() = default;

    // ---- Identity ----
    [[nodiscard]] virtual std::string_view name()      const noexcept = 0;
    [[nodiscard]] virtual DeviceId         device_id() const noexcept = 0;

    // ---- Lifecycle ----
    virtual bool initialize() = 0;
    virtual void finalize()   = 0;

    // ---- Memory ----
    // Allocate `bytes` bytes on this backend's device.
    [[nodiscard]] virtual std::shared_ptr<TensorStorage>
        allocate(size_t bytes, size_t alignment = 64) = 0;

    // Synchronous device-to-device (or host-to-device) copy.
    virtual void transfer(const Tensor& src, Tensor& dst) = 0;

    // Block until all outstanding async work is complete.
    virtual void synchronize() = 0;

    // ---- Kernel dispatch ----
    // Returns true if this backend can handle the given op + dtypes.
    [[nodiscard]] virtual bool
        can_handle(OpType                 op,
                   std::span<const DType> input_dtypes) const = 0;

    // Returns a (potentially cached) kernel for the given op configuration.
    // Must only be called after can_handle() returns true.
    [[nodiscard]] virtual std::unique_ptr<IKernel>
        get_kernel(OpType                 op,
                   std::span<const DType> input_dtypes,
                   const AttributeMap&    attrs) = 0;
};

// ============================================================
// BackendRegistry — global factory registry (singleton).
//
// Usage:
//   // Registering (typically in a backend .cpp file):
//   AXONFORGE_REGISTER_BACKEND("cpu_x86", []{ return std::make_unique<X86Backend>(); });
//
//   // Creating:
//   auto backend = BackendRegistry::instance().create("cpu_x86");
// ============================================================
class BackendRegistry {
public:
    using FactoryFn = std::function<std::unique_ptr<IBackend>()>;

    static BackendRegistry& instance();

    void register_backend(std::string_view id, FactoryFn factory);

    [[nodiscard]] std::unique_ptr<IBackend> create(std::string_view id) const;
    [[nodiscard]] bool has_backend(std::string_view id) const noexcept;
    [[nodiscard]] std::vector<std::string> available_backends() const;

private:
    BackendRegistry()                                = default;
    BackendRegistry(const BackendRegistry&)          = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    std::unordered_map<std::string, FactoryFn> factories_;
};

// ============================================================
// AXONFORGE_REGISTER_BACKEND — convenience wrapper.
// Preferred usage: call BackendRegistry::instance().register_backend()
// from a [[gnu::constructor]] static function in the backend .cpp file.
// Example (see x86_backend.cpp):
//
//   [[gnu::constructor]] static void axonforge_register_my_backend() {
//       ::axonforge::BackendRegistry::instance().register_backend(
//           "my_hw", []{ return std::make_unique<MyBackend>(); });
//   }
// ============================================================

} // namespace axonforge
