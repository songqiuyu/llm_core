#include "axonforge/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace axonforge {

// ---- CpuTensorStorage ----

CpuTensorStorage::CpuTensorStorage(size_t bytes, size_t alignment)
    : bytes_(bytes), owns_(true) {
    if (bytes == 0) { ptr_ = nullptr; return; }
    if (posix_memalign(&ptr_, alignment, bytes) != 0) {
        throw std::bad_alloc();
    }
}

CpuTensorStorage::CpuTensorStorage(void* external_ptr, size_t bytes, bool owns)
    : ptr_(external_ptr), bytes_(bytes), owns_(owns) {}

CpuTensorStorage::~CpuTensorStorage() {
    if (owns_ && ptr_) { free(ptr_); }
}

void*       CpuTensorStorage::data()       noexcept { return ptr_; }
const void* CpuTensorStorage::data() const noexcept { return ptr_; }
size_t      CpuTensorStorage::byte_size()  const noexcept { return bytes_; }
DeviceId    CpuTensorStorage::device()     const noexcept { return DeviceId::cpu(); }

// ---- MemoryArena ----

MemoryArena::MemoryArena(size_t capacity, DeviceId device)
    : capacity_(capacity), device_(device) {
    if (capacity > 0) {
        if (posix_memalign(&buffer_, 64, capacity) != 0) {
            throw std::bad_alloc();
        }
    }
}

MemoryArena::~MemoryArena() {
    if (buffer_) { free(buffer_); }
}

void* MemoryArena::allocate(size_t bytes, size_t alignment) noexcept {
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned_offset + bytes > capacity_) { return nullptr; }
    void* ptr = static_cast<char*>(buffer_) + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void     MemoryArena::reset()    noexcept { offset_ = 0; }
size_t   MemoryArena::used()     const noexcept { return offset_; }
size_t   MemoryArena::capacity() const noexcept { return capacity_; }
DeviceId MemoryArena::device()   const noexcept { return device_; }

// ---- Helpers ----

Strides contiguous_strides(const Shape& shape) {
    Strides s(shape.size());
    if (shape.empty()) return s;
    s.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        s[i] = s[i + 1] * shape[i + 1];
    }
    return s;
}

int64_t numel(const Shape& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ---- Tensor private constructor ----

Tensor::Tensor(std::shared_ptr<TensorStorage> storage,
               Shape   shape,
               Strides strides,
               int64_t offset,
               DType   dtype)
    : storage_(std::move(storage))
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , offset_(offset)
    , dtype_(dtype) {}

// ---- Tensor factory methods ----

Tensor Tensor::empty(Shape shape, DType dtype, DeviceId device) {
    const size_t n     = static_cast<size_t>(::axonforge::numel(shape));
    const size_t bytes = n * dtype_element_size(dtype);
    std::shared_ptr<TensorStorage> storage;
    if (device.is_cpu()) {
        storage = std::make_shared<CpuTensorStorage>(bytes);
    } else {
        throw std::runtime_error("Tensor::empty: device not yet supported");
    }
    return Tensor(std::move(storage), shape, contiguous_strides(shape), 0, dtype);
}

Tensor Tensor::zeros(Shape shape, DType dtype, DeviceId device) {
    auto t = empty(shape, dtype, device);
    if (t.raw_data()) {
        std::memset(t.raw_data(), 0, t.nbytes());
    }
    return t;
}

Tensor Tensor::from_blob(void* data, Shape shape, DType dtype,
                          DeviceId device, bool owns) {
    const size_t bytes = static_cast<size_t>(::axonforge::numel(shape)) * dtype_element_size(dtype);
    auto storage = std::make_shared<CpuTensorStorage>(data, bytes, owns);
    return Tensor(std::move(storage), shape, contiguous_strides(shape), 0, dtype);
}

Tensor Tensor::from_raw_blob(void* data, size_t nbytes,
                               Shape shape, DType dtype,
                               DeviceId /*device*/) {
    auto storage = std::make_shared<CpuTensorStorage>(data, nbytes, /*owns=*/false);
    return Tensor(std::move(storage), shape, contiguous_strides(shape), 0, dtype);
}

// ---- Attributes ----

const Shape&   Tensor::shape()   const noexcept { return shape_; }
const Strides& Tensor::strides() const noexcept { return strides_; }
DType          Tensor::dtype()   const noexcept { return dtype_; }
DeviceId       Tensor::device()  const noexcept {
    return storage_ ? storage_->device() : DeviceId::cpu();
}
int64_t Tensor::ndim()  const noexcept { return static_cast<int64_t>(shape_.size()); }
int64_t Tensor::numel() const noexcept { return ::axonforge::numel(shape_); }
size_t  Tensor::nbytes() const noexcept {
    return static_cast<size_t>(numel()) * dtype_element_size(dtype_);
}
int64_t Tensor::dim(int i) const {
    if (i < 0) i += static_cast<int>(shape_.size());
    return shape_.at(static_cast<size_t>(i));
}
bool Tensor::is_valid() const noexcept { return storage_ != nullptr; }

bool Tensor::is_contiguous() const noexcept {
    if (shape_.empty()) return true;
    int64_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return offset_ == 0;
}

// ---- View operations (O(1)) ----

Tensor Tensor::reshape(Shape new_shape) const {
    if (!is_contiguous()) {
        return contiguous().reshape(std::move(new_shape));
    }
    return Tensor(storage_, std::move(new_shape),
                  contiguous_strides(new_shape), offset_, dtype_);
}

Tensor Tensor::permute(std::vector<int> perm) const {
    const int n = static_cast<int>(shape_.size());
    Shape   new_shape(n);
    Strides new_strides(n);
    for (int i = 0; i < n; ++i) {
        int p = perm[i];
        if (p < 0) p += n;
        new_shape[i]   = shape_[p];
        new_strides[i] = strides_[p];
    }
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), offset_, dtype_);
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t /*step*/) const {
    if (dim < 0) dim += static_cast<int>(shape_.size());
    Shape new_shape = shape_;
    new_shape[dim]  = end - start;
    int64_t new_offset = offset_ + start * strides_[dim];
    return Tensor(storage_, std::move(new_shape), strides_, new_offset, dtype_);
}

Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += static_cast<int>(shape_.size());
    if (shape_[dim] != 1) return *this;
    Shape   s = shape_;   s.erase(s.begin() + dim);
    Strides st = strides_; st.erase(st.begin() + dim);
    return Tensor(storage_, std::move(s), std::move(st), offset_, dtype_);
}

Tensor Tensor::unsqueeze(int dim) const {
    Shape   s  = shape_;
    Strides st = strides_;
    if (dim < 0) dim += static_cast<int>(s.size()) + 1;
    s.insert(s.begin() + dim, 1);
    int64_t new_stride = (dim < static_cast<int>(st.size()))
                       ? st[dim] * s[dim + 1]
                       : 1;
    st.insert(st.begin() + dim, new_stride);
    return Tensor(storage_, std::move(s), std::move(st), offset_, dtype_);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    const int n = static_cast<int>(shape_.size());
    if (start_dim < 0) start_dim += n;
    if (end_dim   < 0) end_dim   += n;
    int64_t flat = 1;
    for (int i = start_dim; i <= end_dim; ++i) flat *= shape_[i];
    Shape new_shape;
    for (int i = 0; i < start_dim; ++i)  new_shape.push_back(shape_[i]);
    new_shape.push_back(flat);
    for (int i = end_dim + 1; i < n; ++i) new_shape.push_back(shape_[i]);
    return reshape(std::move(new_shape));
}

// ---- Data-copying operations ----

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    auto out = Tensor::empty(shape_, dtype_, device());
    // TODO: implement non-contiguous copy with stride iteration
    throw std::runtime_error("Tensor::contiguous: non-contiguous copy not yet implemented");
    return out;
}

Tensor Tensor::to(DeviceId /*target*/) const {
    // TODO(Phase2): cross-device copy via backend transfer
    throw std::runtime_error("Tensor::to(DeviceId): not yet implemented");
}

Tensor Tensor::to(DType /*target_dtype*/) const {
    // TODO(Phase1): quantize / dequantize
    throw std::runtime_error("Tensor::to(DType): not yet implemented");
}

// ---- Raw data access ----

void*       Tensor::raw_data()       noexcept { return storage_ ? storage_->data() : nullptr; }
const void* Tensor::raw_data() const noexcept { return storage_ ? storage_->data() : nullptr; }

std::shared_ptr<TensorStorage> Tensor::storage() const noexcept { return storage_; }
int64_t Tensor::storage_offset() const noexcept { return offset_; }

} // namespace axonforge
