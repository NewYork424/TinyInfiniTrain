#include <iostream>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/autograd/matmul.h"

using namespace infini_train;

TEST(MatmulTest, BasicMatrixMultiply) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));
    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 9, 10, 11, 12};
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});

    // 验证输出结果 (2x2)
    float expected[] = {58, 64, 
                        139, 154};
    for (int i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BatchedMatrixMultiply) {
    // 创建输入张量 (2, 2, 3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));

    // 创建权重张量 (2, 3, 2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    // 执行正向传播
    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});

    // 验证输出结果 (2, 2, 2)
    float expected[] = {22, 28, 49, 64,    // batch 0
                        220, 244, 301, 334}; // batch 1
    for (int i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BackwardPass) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 
                          4, 5, 6};
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));

    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 
                          9, 10, 
                          11, 12};
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    // 模拟梯度输出 (2x2)
    auto grad_output = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* grad_data = static_cast<float*>(grad_output->DataPtr());
    float grad_values[] = {0.1, 0.2, 
                         0.3, 0.4};
    std::memcpy(grad_data, grad_values, std::size(grad_values) * sizeof(float));

    auto output_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);

    autograd::Matmul matmul_op;
    matmul_op.SetupContext({input, other}, {output_tensor});
    auto output = matmul_op.Backward({grad_output});

    // 验证输入梯度 (2x3)make
    float expected_grad_input[] = {2.3, 2.9, 3.5, 
                                   5.3, 6.7, 8.1};
    for (int i = 0; i < std::size(expected_grad_input); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected_grad_input[i]);
    }

    // 验证权重梯度 (3x2)
    float expected_grad_other[] = {1.3, 1.8, 
                                   1.7, 2.4, 
                                   2.1, 3.0};
    for (int i = 0; i < std::size(expected_grad_other); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[1]->DataPtr())[i], expected_grad_other[i]);
    }
}
