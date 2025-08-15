#include <cstddef>
#include <memory>
#include <cmath>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:https://zhuanlan.zhihu.com/p/32698042
    // =================================== 作业 ===================================

    // 获取梯度、参数、动量和二阶动量的指针
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 获取元素数量
    int64_t num_elements = grad->NumElements();

    // 计算偏置修正系数
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float lr_t = learning_rate * std::sqrt(1 - beta2_t) / (1 - beta1_t);

    // 更新参数
    for (int64_t idx = 0; idx < num_elements; ++idx) {
        // 更新一阶动量 m
        m_ptr[idx] = beta1 * m_ptr[idx] + (1 - beta1) * grad_ptr[idx];

        // 更新二阶动量 v
        v_ptr[idx] = beta2 * v_ptr[idx] + (1 - beta2) * grad_ptr[idx] * grad_ptr[idx];

        // 更新参数
        param_ptr[idx] -= lr_t * m_ptr[idx] / (std::sqrt(v_ptr[idx]) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
