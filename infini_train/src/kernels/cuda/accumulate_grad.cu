#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

// 前向声明 Adam kernel
__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                         float lr_t, float beta1, float beta2, float eps, size_t num_elements);

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    // 获取梯度、参数、动量和二阶动量的指针
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 获取元素数量
    size_t num_elements = grad->NumElements();

    // 计算偏置修正系数
    float beta1_t = powf(beta1, t);
    float beta2_t = powf(beta2, t);
    float lr_t = learning_rate * sqrtf(1 - beta2_t) / (1 - beta1_t);

    // CUDA 核函数
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, lr_t, beta1, beta2, eps, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                         float lr_t, float beta1, float beta2, float eps, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // 更新一阶动量 m
        m_ptr[idx] = beta1 * m_ptr[idx] + (1 - beta1) * grad_ptr[idx];

        // 更新二阶动量 v
        v_ptr[idx] = beta2 * v_ptr[idx] + (1 - beta2) * grad_ptr[idx] * grad_ptr[idx];

        // 更新参数
        param_ptr[idx] -= lr_t * m_ptr[idx] / (sqrtf(v_ptr[idx]) + eps);
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
