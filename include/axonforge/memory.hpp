#pragma once
#include "axonforge/common.hpp"
#include <cstddef>
#include <memory>

namespace axonforge {

// ============================================================
// TensorStorage — abstract backing memory for a Tensor.
// Concrete implementations: CpuTensorStorage, CudaTensorStorage, etc.
// ============================================================
class TensorStorage {
public:
    virtual ~TensorStorage() = default;

    [[nodiscard]] virtual void*       data()      noexcept = 0;
    [[nodiscard]] virtual const void* data() const noexcept = 0;
    [[nodiscard]] virtual size_t      byte_size() const noexcept = 0;
    [[nodiscard]] virtual DeviceId    device()    const noexcept = 0;

    [[nodiscard]] bool is_cpu() const noexcept {
        return device().type == DeviceType::CPU;
    }
};

// ============================================================
// CpuTensorStorage — 64-byte aligned heap allocation on CPU.
// ============================================================
class CpuTensorStorage final : public TensorStorage {
public:
    // Allocate `bytes` bytes with `alignment`-byte alignment.
    explicit CpuTensorStorage(size_t bytes, size_t alignment = 64);

    // Wrap an externally managed pointer (no ownership transfer if owns=false).
    CpuTensorStorage(void* external_ptr, size_t bytes, bool owns = false);

    ~CpuTensorStorage() override;

    // Non-copyable, movable
    CpuTensorStorage(const CpuTensorStorage&)            = delete;
    CpuTensorStorage& operator=(const CpuTensorStorage&) = delete;
    CpuTensorStorage(CpuTensorStorage&&)                 = default;

    [[nodiscard]] void*       data()      noexcept override;
    [[nodiscard]] const void* data() const noexcept override;
    [[nodiscard]] size_t      byte_size() const noexcept override;
    [[nodiscard]] DeviceId    device()    const noexcept override;

private:
    void*  ptr_{nullptr};
    size_t bytes_{0};
    bool   owns_{true};
};

// ============================================================
// MemoryArena — bump-pointer arena for temporary tensors.
//
// Designed to be reset between forward passes; does NOT release
// the underlying buffer on reset() — only on destruction.
// Basis for the MemoryPlanner's "liveness-aware slot reuse".
// ============================================================
class MemoryArena {
public:
    explicit MemoryArena(size_t capacity, DeviceId device = DeviceId::cpu());
    ~MemoryArena();

    MemoryArena(const MemoryArena&)            = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    // Allocate `bytes` bytes aligned to `alignment`.
    // Returns nullptr if out of space.
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = 64) noexcept;

    // Reset the arena (free all previously allocated blocks logically).
    // Does NOT free the backing buffer.
    void reset() noexcept;

    [[nodiscard]] size_t   used()     const noexcept;
    [[nodiscard]] size_t   capacity() const noexcept;
    [[nodiscard]] DeviceId device()   const noexcept;

private:
    void*    buffer_{nullptr};
    size_t   capacity_{0};
    size_t   offset_{0};
    DeviceId device_;
};

} // namespace axonforge
