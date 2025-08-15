#include "cuda_runtime.h"
#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void TrilForwardKernel(const float *input, float *output, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    int64_t rows = input->Dims()[0];
    int64_t cols = input->Dims()[1];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TrilForwardKernel<<<num_blocks, threads_per_block>>>(static_cast<float *>(input->DataPtr()),
                                                         static_cast<float *>(output->DataPtr()), rows, cols, diagonal);
    return output;
}

__global__ void TrilBackwardKernel(const float *grad_output, float *grad_input, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        grad_input[idx] = grad_output[idx];
    } else {
        grad_input[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    int rows = grad_output->Dims()[0];
    int cols = grad_output->Dims()[1];

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TrilBackwardKernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                          static_cast<float *>(grad_input->DataPtr()), rows, cols,
                                                          diagonal);

    return grad_input;
}

__global__ void TriuForwardKernel(const float *input, float *output, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal <= 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TriuForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    int64_t rows = input->Dims()[0];
    int64_t cols = input->Dims()[1];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TriuForwardKernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(input->DataPtr()),
                                                         static_cast<float *>(output->DataPtr()), rows, cols, diagonal);

    return output;
}

__global__ void TriuBackwardKernel(const float *grad_output, float *grad_input, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal <= 0) {
        grad_input[idx] = grad_output[idx];
    } else {
        grad_input[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TriuBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    int rows = grad_output->Dims()[0];
    int cols = grad_output->Dims()[1];

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TriuBackwardKernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                          static_cast<float *>(grad_input->DataPtr()), rows, cols,
                                                          diagonal);

    return grad_input;
}

__global__ void TransposeForwardKernel(const float *input, float *output, const int64_t *in_dims,
                                       const int64_t *in_strides, const int64_t *out_strides, int64_t ndim,
                                       int64_t dim0, int64_t dim1, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t remaining = idx;
    int64_t coords[8];

    // 1. decode coord from output index
    for (int i = 0; i < ndim; ++i) {
        coords[i] = remaining / out_strides[i];
        remaining %= out_strides[i];
    }

    // 2. swap the coordinates
    int64_t tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    // 3. compute input flat index
    int64_t in_flat_idx = 0;
    for (int i = 0; i < ndim; ++i) { in_flat_idx += coords[i] * in_strides[i]; }

    output[idx] = input[in_flat_idx];
}

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1) {
    CHECK_LE(input->Dims().size(), 8);
    dim0 = dim0 < 0 ? dim0 + input->Dims().size() : dim0;
    dim1 = dim1 < 0 ? dim1 + input->Dims().size() : dim1;
    CHECK(dim0 >= 0 && dim0 < input->Dims().size() && dim1 >= 0 && dim1 < input->Dims().size());

    auto in_dims = input->Dims();
    std::vector<int64_t> out_dims = in_dims;
    std::swap(out_dims[dim0], out_dims[dim1]);

    auto output = std::make_shared<Tensor>(out_dims, input->Dtype(), input->GetDevice());
    output->Fill<float>(0.0f);
    int64_t ndim = in_dims.size();
    int64_t num_elements = output->NumElements();

    // compute strides of in_dims and out_dims
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    // Allocate device memory for dims and strides
    int64_t *device_buffer;
    cudaMallocAsync(&device_buffer, 3 * ndim * sizeof(int64_t), 0);

    int64_t *in_dims_dev = device_buffer;
    int64_t *in_strides_dev = device_buffer + ndim;
    int64_t *out_strides_dev = device_buffer + 2 * ndim;

    std::vector<int64_t> host_buffer;
    host_buffer.insert(host_buffer.end(), in_dims.begin(), in_dims.end());
    host_buffer.insert(host_buffer.end(), in_strides.begin(), in_strides.end());
    host_buffer.insert(host_buffer.end(), out_strides.begin(), out_strides.end());

    cudaMemcpyAsync(device_buffer, host_buffer.data(), 3 * ndim * sizeof(int64_t), cudaMemcpyHostToDevice, 0);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    TransposeForwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(output->DataPtr()), in_dims_dev,
        in_strides_dev, out_strides_dev, ndim, dim0, dim1, num_elements);

    cudaFreeAsync(device_buffer, 0);

    return output;
}

std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1) {
    return TransposeForward(grad_output, dim1, dim0);
}

__global__ void MaskForwardKernel(const float *input, const float *mask, float *output, float value, int batch_size,
                                  int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        output[i] = (mask[i % mask_size] == 1.0f) ? value : input[i];
    }
}

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value) {
    auto input_shape = input->Dims();
    auto mask_shape = mask->Dims();
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(mask->Dtype()));

    int64_t input_dims = input_shape.size();
    int64_t mask_dims = mask_shape.size();
    for (int i = 0; i < mask_dims; ++i) {
        int input_dim = input_shape[input_dims - mask_dims + i];
        int mask_dim = mask_shape[i];
        CHECK(input_dim == mask_dim || mask_dim == 1);
    }

    int64_t mask_size = mask->NumElements();
    int64_t batch_size = input->NumElements() / mask_size;

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (input->NumElements() + threads_per_block - 1) / threads_per_block;

    MaskForwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(mask->DataPtr()),
        static_cast<float *>(output->DataPtr()), value, batch_size, mask_size);
    return output;
}

__global__ void MaskBackwardKernel(const float *grad_output, const float *mask, float *grad_input, int batch_size,
                                   int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        grad_input[i] = (mask[i % mask_size] == 1.0f) ? 0.0f : grad_output[i];
    }
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    auto output_shape = grad_output->Dims();
    auto mask_shape = mask->Dims();
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(mask->Dtype()));

    int64_t output_dims = output_shape.size();
    int64_t mask_dims = mask_shape.size();
    for (int i = 0; i < mask_dims; ++i) {
        int out_dim = output_shape[output_dims - mask_dims + i];
        int mask_dim = mask_shape[i];
        CHECK(out_dim == mask_dim || mask_dim == 1);
    }

    int64_t mask_size = mask->NumElements();
    int64_t batch_size = grad_output->NumElements() / mask_size;

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = (grad_output->NumElements() + threads_per_block - 1) / threads_per_block;

    MaskBackwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<const float *>(mask->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), batch_size, mask_size);
    return grad_input;
}

__global__ void RepeatInterleaveForwardKernel(const float *input, float *output, int64_t outer, int64_t dim_size,
                                              int64_t inner, int64_t repeat) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim_size * repeat * inner;
    if (idx >= total) {
        return;
    }

    int64_t i = idx / inner;
    int64_t j = idx % inner;

    int64_t o = i / (dim_size * repeat);
    int64_t di = (i / repeat) % dim_size;

    output[idx] = input[(o * dim_size + di) * inner + j];
}

std::shared_ptr<Tensor> RepeatInterleaveForward(const std::shared_ptr<Tensor> &input, int64_t repeat, int64_t dim) {
    CHECK_GT(repeat, 0);
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input->Dims().size());

    const auto &input_dims = input->Dims();
    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    std::vector<int64_t> output_dims = input_dims;
    output_dims[dim] = dim_size * repeat;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    int64_t total_elements = outer * dim_size * repeat * inner;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    RepeatInterleaveForwardKernel<<<num_blocks, threads_per_block>>>(input_ptr, output_ptr, outer, dim_size, inner,
                                                                     repeat);

    return output;
}

__global__ void RepeatInterleaveBackwardKernel(const float *grad_output, float *grad_input, int64_t outer,
                                               int64_t dim_size, int64_t inner, int64_t repeat) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim_size * inner;
    if (idx >= total) {
        return;
    }

    int64_t i = idx / inner;
    int64_t j = idx % inner;

    int64_t o = i / dim_size;
    int64_t di = i % dim_size;

    float sum = 0.0f;
    for (int64_t r = 0; r < repeat; ++r) {
        int64_t out_idx = ((o * dim_size * repeat + di * repeat + r) * inner) + j;
        sum += grad_output[out_idx];
    }
    grad_input[idx] = sum;
}

std::shared_ptr<Tensor> RepeatInterleaveBackward(const std::shared_ptr<Tensor> &grad_output,
                                                 const std::vector<int64_t> &input_dims, int64_t dim) {
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input_dims.size());

    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    int64_t repeat = grad_output->Dims()[dim] / dim_size;
    CHECK_EQ(grad_output->Dims()[dim], dim_size * repeat);

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    const float *grad_out_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_in_ptr = static_cast<float *>(grad_input->DataPtr());

    int64_t total_elements = outer * dim_size * inner;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    RepeatInterleaveBackwardKernel<<<num_blocks, threads_per_block>>>(grad_out_ptr, grad_in_ptr, outer, dim_size, inner,
                                                                      repeat);

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_TRANSFORM_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_TRANSFORM_KERNEL(TrilForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TrilBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(TriuForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TriuBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(TransposeForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TransposeBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(MaskForward)
REGISTER_CUDA_TRANSFORM_KERNEL(MaskBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(RepeatInterleaveForward)
REGISTER_CUDA_TRANSFORM_KERNEL(RepeatInterleaveBackward)

#undef REGISTER_CUDA_TRANSFORM_KERNEL
