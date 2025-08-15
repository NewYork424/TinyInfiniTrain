#include<iostream>
#include<vector>

#include "gtest/gtest.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/autograd/elementwise.h"

using namespace infini_train;

TEST(ElementwiseTest, NegForward) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{3},  // dims
        DataType::kFLOAT32,      // dtype
        Device(DeviceType::kCPU, 0) // device
    );
    
    float* data = static_cast<float*>(input->DataPtr());
    data[0] = 1.0f;
    data[1] = -2.0f;
    data[2] = 0.0f;

    autograd::Neg neg_op;
    auto outputs = neg_op.Forward({input});
    ASSERT_EQ(outputs.size(), 1);

    // 预期输出：[-1.0, 2.0, 0.0]
    std::vector<float> expected = {-1.0f, 2.0f, 0.0f};
    const float* result_data = static_cast<const float*>(outputs[0]->DataPtr());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST(ElementwiseTest, NegBackward) {
    auto grad_output = std::make_shared<Tensor>(
        std::vector<int64_t>{3},  // dims
        DataType::kFLOAT32,       // dtype
        Device(DeviceType::kCPU, 0) // device
    );

    float* grad_data = static_cast<float*>(grad_output->DataPtr());
    grad_data[0] = 1.0f;
    grad_data[1] = 1.0f;
    grad_data[2] = 1.0f;

    autograd::Neg neg_op;
    auto grad_inputs = neg_op.Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 1);

    // 预期梯度：[-1.0, -1.0, -1.0]
    std::vector<float> expected = {-1.0f, -1.0f, -1.0f};
    const float* result_data = static_cast<const float*>(grad_inputs[0]->DataPtr());
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}
