#include <algorithm>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
__global__ void SplitForwardKernel(const float *input, float *output, int64_t N, int64_t H_in, int64_t H_out, int64_t W,
                                   int64_t start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W;

    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H_out;
        int n = idx / (H_out * W);

        int input_h = h + start_idx;
        int input_idx = n * H_in * W + input_h * W + w;
        int output_idx = n * H_out * W + h * W + w;

        output[output_idx] = input[input_idx];
    }
}

std::vector<std::shared_ptr<Tensor>> SplitForward(const std::shared_ptr<Tensor> &input, int64_t split_size, int dim) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    const auto &input_dims = input->Dims();
    CHECK_LT(dim, input_dims.size());

    std::vector<std::shared_ptr<Tensor>> outputs;

    const int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t W = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t H_in = input_dims[dim];

    for (int64_t start = 0; start < H_in; start += split_size) {
        auto output_dims = input_dims;
        const int64_t H_out = std::min(split_size, H_in - start);
        output_dims[dim] = H_out;

        auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

        int64_t total = N * H_out * W;
        int threads_per_block = 256;
        int num_blocks = (total + threads_per_block - 1) / threads_per_block;

        SplitForwardKernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(input->DataPtr()),
                                                              static_cast<float *>(output->DataPtr()), N, H_in, H_out,
                                                              W, start);
        outputs.push_back(std::move(output));
    }

    return outputs;
}

__global__ void SplitBackwardKernel(const float *const *grad_outputs, float *grad_input, int64_t N, int64_t H_in,
                                    int64_t W, int64_t split_size, int64_t num_splits, const int64_t *H_outs) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * H_in * W;
    if (idx >= total) {
        return;
    }

    int64_t w = idx % W;
    int64_t h = (idx / W) % H_in;
    int64_t n = idx / (H_in * W);

    int64_t split_idx = h / split_size;
    if (split_idx >= num_splits) {
        return;
    }

    int64_t H_out = H_outs[split_idx];
    int64_t local_h = h - split_idx * split_size;

    if (local_h >= H_out) {
        return;
    }

    const float *grad_output = grad_outputs[split_idx];
    float value = grad_output[(n * H_out + local_h) * W + w];
    grad_input[(n * H_in + h) * W + w] = value;
}

std::shared_ptr<Tensor> SplitBackward(const std::vector<int64_t> &input_dims, int64_t split_size, int dim,
                                      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    CHECK_LT(dim, input_dims.size());

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_outputs[0]->GetDevice());
    grad_input->Fill<float>(0.0f);
    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    int64_t W = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    int64_t H_in = input_dims[dim];
    int64_t num_splits = grad_outputs.size();

    // init the array of grad_output ptrs
    std::vector<const float *> host_grad_output_ptrs;
    for (const auto &grad_output : grad_outputs) {
        host_grad_output_ptrs.push_back(static_cast<const float *>(grad_output->DataPtr()));
    }

    void *device_ptr;
    const float **device_grad_output_ptrs;
    int64_t *device_H_outs;
    cudaMallocAsync(&device_ptr, (sizeof(float *) + sizeof(int64_t)) * num_splits, 0);
    device_grad_output_ptrs = (const float **)(device_ptr);
    device_H_outs = reinterpret_cast<int64_t *>(device_grad_output_ptrs + num_splits);

    cudaMemcpyAsync(device_grad_output_ptrs, host_grad_output_ptrs.data(), sizeof(float *) * num_splits,
                    cudaMemcpyHostToDevice, 0);

    // init H_out for each split
    std::vector<int64_t> H_outs(num_splits);
    for (int i = 0; i < num_splits; ++i) { H_outs[i] = std::min(split_size, H_in - i * split_size); }

    cudaMemcpyAsync(device_H_outs, H_outs.data(), sizeof(int64_t) * num_splits, cudaMemcpyHostToDevice, 0);

    int64_t total_elements = N * H_in * W;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    SplitBackwardKernel<<<num_blocks, threads_per_block>>>(device_grad_output_ptrs,
                                                           static_cast<float *>(grad_input->DataPtr()), N, H_in, W,
                                                           split_size, num_splits, device_H_outs);

    cudaFreeAsync(device_ptr, 0);
    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SPLIT_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SPLIT_KERNEL(SplitForward)
REGISTER_CUDA_SPLIT_KERNEL(SplitBackward)

#undef REGISTER_CUDA_SPLIT_KERNEL
