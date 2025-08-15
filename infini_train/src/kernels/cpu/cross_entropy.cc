#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &target) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int num_classes = *input_dims.rbegin();

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    static_cast<float *>(output->DataPtr())[0] = 0.0f;
    for (int64_t i = 0; i < bs; ++i) {
        float max_logit = kNegativeInfinity;
        for (int64_t j = 0; j < num_classes; ++j) {
            max_logit = std::max(max_logit, static_cast<const float *>(input->DataPtr())[i * num_classes + j]);
        }
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes; ++j) {
            sum_exp += exp(static_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
        }
        if (target->Dtype() == DataType::kUINT8) {
            static_cast<float *>(output->DataPtr())[0]
                -= log(exp(static_cast<const float *>(
                               input->DataPtr())[i * num_classes + static_cast<const uint8_t *>(target->DataPtr())[i]]
                           - max_logit)
                       / sum_exp);
        } else if (target->Dtype() == DataType::kINT64) {
            static_cast<float *>(output->DataPtr())[0]
                -= log(exp(static_cast<const float *>(
                               input->DataPtr())[i * num_classes + static_cast<const int64_t *>(target->DataPtr())[i]]
                           - max_logit)
                       / sum_exp);
        } else {
            LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target->Dtype());
        }
    }
    static_cast<float *>(output->DataPtr())[0] /= bs;
    return {output};
}

std::shared_ptr<Tensor> CrossEntropyBackward(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &target,
                                             const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int num_classes = *input_dims.rbegin();

    CHECK_EQ(grad_output->Dims().size(), 0);
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    std::vector<float> softmax(bs * num_classes, 0.0f);
    for (int64_t i = 0; i < bs; ++i) {
        float max_logit = kNegativeInfinity;
        for (int64_t j = 0; j < num_classes; ++j) {
            max_logit = std::max(max_logit, static_cast<const float *>(input->DataPtr())[i * num_classes + j]);
        }
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes; ++j) {
            sum_exp += exp(static_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
        }
        for (int64_t j = 0; j < num_classes; ++j) {
            const auto idx = i * num_classes + j;
            softmax[idx] = exp(static_cast<const float *>(input->DataPtr())[idx] - max_logit) / sum_exp;
        }
    }
    for (int64_t i = 0; i < bs; ++i) {
        auto target_idx = 0;
        if (target->Dtype() == DataType::kUINT8) {
            target_idx = static_cast<const uint8_t *>(target->DataPtr())[i];
        } else if (target->Dtype() == DataType::kINT64) {
            target_idx = static_cast<const int64_t *>(target->DataPtr())[i];
        } else {
            LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target->Dtype());
        }
        for (int64_t j = 0; j < num_classes; ++j) {
            const auto idx = i * num_classes + j;
            static_cast<float *>(grad_input->DataPtr())[idx] = static_cast<const float *>(grad_output->DataPtr())[0]
                                                             * (softmax[idx] - (j == target_idx ? 1.0f : 0.0f)) / bs;
        }
    }
    return {grad_input};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CROSS_ENTROPY_KERNEL(kernel_name)                                                                 \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CROSS_ENTROPY_KERNEL(CrossEntropyForward)
REGISTER_CPU_CROSS_ENTROPY_KERNEL(CrossEntropyBackward)

#undef REGISTER_CPU_CROSS_ENTROPY_KERNEL
