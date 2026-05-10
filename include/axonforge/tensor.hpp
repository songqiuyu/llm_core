#pragma once
#include "axonforge/common.hpp"
#include "axonforge/dtype.hpp"
#include "axonforge/memory.hpp"
#include <vector>
#include <span>
#include <memory>
#include <cstdint>

namespace axonforge {

using Shape   = std::vector<int64_t>;
using Strides = std::vector<int64_t>;  // in units of elements

// ============================================================
// Tensor — N-dimensional array with view semantics.
//
// A Tensor is a lightweight value type (cheap to copy/move):
//   - shape + strides + offset describe the view
//   - storage is shared via shared_ptr (ref-counted)
//
// View operations (reshape, permute, slice) are O(1) — they
// produce a new Tensor sharing the same TensorStorage.
// Actual data copy only happens via .contiguous() or .to().
// ============================================================
class Tensor {
public:
    // ---- Construction ----

    Tensor() = default;

    // Allocate uninitialised contiguous tensor on `device`.
    [[nodiscard]] static Tensor empty(Shape shape, DType dtype,
                                      DeviceId device = DeviceId::cpu());

    // Allocate and zero-initialise.
    [[nodiscard]] static Tensor zeros(Shape shape, DType dtype,
                                      DeviceId device = DeviceId::cpu());

    // Wrap a raw pointer (no copy). If owns=false, caller manages lifetime.
    [[nodiscard]] static Tensor from_blob(void* data, Shape shape,
                                          DType dtype,
                                          DeviceId device = DeviceId::cpu(),
                                          bool owns = false);

    // Wrap a raw pointer with an explicit byte count.
    // Use for block-quantized types where nbytes != numel * dtype_element_size().
    // The pointer must remain valid for the Tensor's lifetime (owns=false).
    [[nodiscard]] static Tensor from_raw_blob(void* data, size_t nbytes,
                                              Shape shape, DType dtype,
                                              DeviceId device = DeviceId::cpu());

    // ---- Copy / move (cheap — only copies view metadata + increments refcount) ----
    Tensor(const Tensor&)            = default;
    Tensor(Tensor&&)                 = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&)      = default;
    ~Tensor()                        = default;

    // ---- Attributes ----

    [[nodiscard]] const Shape&   shape()   const noexcept;
    [[nodiscard]] const Strides& strides() const noexcept;
    [[nodiscard]] DType          dtype()   const noexcept;
    [[nodiscard]] DeviceId       device()  const noexcept;
    [[nodiscard]] int64_t        ndim()    const noexcept;
    [[nodiscard]] int64_t        numel()   const noexcept;
    [[nodiscard]] size_t         nbytes()  const noexcept;
    [[nodiscard]] int64_t        dim(int i) const;
    [[nodiscard]] bool           is_contiguous() const noexcept;
    [[nodiscard]] bool           is_valid()      const noexcept;

    // ---- O(1) view operations (no data copy, share storage) ----

    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    [[nodiscard]] Tensor permute(std::vector<int> perm) const;
    [[nodiscard]] Tensor slice(int dim, int64_t start, int64_t end,
                               int64_t step = 1) const;
    [[nodiscard]] Tensor squeeze(int dim)   const;
    [[nodiscard]] Tensor unsqueeze(int dim) const;
    [[nodiscard]] Tensor flatten(int start_dim = 0, int end_dim = -1) const;

    // ---- Data-copying operations ----

    // Returns a contiguous copy (if already contiguous, returns *this).
    [[nodiscard]] Tensor contiguous() const;

    // Returns a copy on `target_device`.
    [[nodiscard]] Tensor to(DeviceId target_device) const;

    // Returns a type-cast copy.
    [[nodiscard]] Tensor to(DType target_dtype) const;

    // ---- Raw data access (CPU contiguous only) ----

    [[nodiscard]] void*       raw_data()       noexcept;
    [[nodiscard]] const void* raw_data() const noexcept;

    template<typename T>
    [[nodiscard]] std::span<T> data_as() noexcept {
        return {static_cast<T*>(raw_data()), static_cast<size_t>(numel())};
    }

    template<typename T>
    [[nodiscard]] std::span<const T> data_as() const noexcept {
        return {static_cast<const T*>(raw_data()), static_cast<size_t>(numel())};
    }

    // ---- Storage access (for backend implementations) ----
    [[nodiscard]] std::shared_ptr<TensorStorage> storage() const noexcept;
    [[nodiscard]] int64_t storage_offset() const noexcept;  // in elements

private:
    // Private constructor used by view ops and factory methods.
    Tensor(std::shared_ptr<TensorStorage> storage,
           Shape   shape,
           Strides strides,
           int64_t offset,
           DType   dtype);

    std::shared_ptr<TensorStorage> storage_;
    Shape   shape_;
    Strides strides_;
    int64_t offset_{0};
    DType   dtype_{DType::UNKNOWN};
};

// ---- Helpers ----

// Compute contiguous (row-major) strides for a given shape.
[[nodiscard]] Strides contiguous_strides(const Shape& shape);

// Total number of elements.
[[nodiscard]] int64_t numel(const Shape& shape);

} // namespace axonforge
