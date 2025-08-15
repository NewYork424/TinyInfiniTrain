#include <cmath>
#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                 const std::shared_ptr<Tensor> &bias, const float eps) {
    /*
        x: [bs, seq_len, embed_dim]
        -> LayerNorm (w: [embed_dim], b: [embed_dim])
        -> o: [bs, seq_len, embed_dim]
    */
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen, embed_dim}, DataType::kFLOAT32);
    auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32);
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32);
    mean->Fill<float>(0.0f);
    rstd->Fill<float>(0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < max_seqlen; t++) {
            float m = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                m += static_cast<float *>(input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
            }
            m = m / embed_dim;

            float v = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                float xshift
                    = static_cast<float *>(input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i] - m;
                v += xshift * xshift;
            }
            v = v / embed_dim;

            float s = 1.0f / sqrtf(v + eps);

            for (int i = 0; i < embed_dim; i++) {
                float n = (s
                           * (static_cast<float *>(input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i]
                              - m)); // normalize
                float o = n * static_cast<float *>(weight->DataPtr())[i]
                        + static_cast<float *>(bias->DataPtr())[i]; // scale and shift
                static_cast<float *>(output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            static_cast<float *>(mean->DataPtr())[b * max_seqlen + t] = m;
            static_cast<float *>(rstd->DataPtr())[b * max_seqlen + t] = s;
        }
    }

    return {output, mean, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);
    CHECK_NE(mean, nullptr);
    CHECK_NE(rstd, nullptr);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32);
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32);

    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    grad_bias->Fill<float>(0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < max_seqlen; t++) {
            float mean_bt = static_cast<float *>(mean->DataPtr())[b * max_seqlen + t];
            float rstd_bt = static_cast<float *>(rstd->DataPtr())[b * max_seqlen + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                float norm_bti
                    = (static_cast<float *>(input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i] - mean_bt)
                    * rstd_bt;
                float dnorm_i
                    = static_cast<float *>(weight->DataPtr())[i]
                    * static_cast<float *>(grad_output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / embed_dim;
            dnorm_norm_mean = dnorm_norm_mean / embed_dim;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < embed_dim; i++) {
                float norm_bti
                    = (static_cast<float *>(input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i] - mean_bt)
                    * rstd_bt;
                float dnorm_i
                    = static_cast<float *>(weight->DataPtr())[i]
                    * static_cast<float *>(grad_output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
                // gradient contribution to bias
                static_cast<float *>(grad_bias->DataPtr())[i]
                    += static_cast<float *>(grad_output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
                // gradient contribution to weight
                static_cast<float *>(grad_weight->DataPtr())[i]
                    += norm_bti
                     * static_cast<float *>(grad_output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i;                    // term 1
                dval -= dnorm_mean;                 // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt;                    // final scale
                static_cast<float *>(grad_input->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i] += dval;
            }
        }
    }
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LAYERNORM_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_CPU_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_CPU_LAYERNORM_KERNEL
