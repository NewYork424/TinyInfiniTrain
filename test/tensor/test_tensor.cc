#include<iostream>
#include<vector>

#include "gtest/gtest.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"

using namespace infini_train;

TEST(TensorAutogradTest, BackwardComputesGradient) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(2.0f);

    auto y = x->Pow(2); 

    y->Backward();

    // dy/dx = 2x = [4.0] (当x=2时)
    float* grad = static_cast<float*>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(grad[0], 4.0f);
}

TEST(TensorAutogradTest, BackwardWithMultipleOutputs) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(1.0f);  // [1.0, 1.0, 1.0]

    auto y1 = x->Mul(2.0f);  // [2.0, 2.0, 2.0]
    auto y2 = x->Pow(3);     // [1.0, 1.0, 1.0]

    auto grad1 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    grad1->Fill(1.0f);
    auto grad2 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    grad2->Fill(2.0f);
    y1->Backward(grad1);
    y2->Backward(grad2);

    // dy1/dx = 2, dy2/dx = 3x^2 = 3  总梯度 = 1 * 2 + 2 * 3 = 8
    float* grad = static_cast<float*>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(grad[0], 8.0f);
    EXPECT_FLOAT_EQ(grad[1], 8.0f);
    EXPECT_FLOAT_EQ(grad[2], 8.0f);
}

TEST(TensorTransformTest, Flatten2DTo1D) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32);
    float* data = static_cast<float*>(t->DataPtr());
    for (int i = 0; i < 12; ++i) {
        data[i] = i + 1;  // 1-12
    }

    auto flattened = t->Flatten(0, 1);

    EXPECT_EQ(flattened->Dims(), std::vector<int64_t>({12}));
    float* flat_data = static_cast<float*>(flattened->DataPtr());
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(flat_data[i], i + 1);
    }
}

TEST(TensorTransformTest, FlattenWithRange) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32);

    auto result = t->Flatten(1, -1);
    
    EXPECT_EQ(result->Dims(), (std::vector<int64_t>{2, 12}));
}

TEST(TensorTransformTest, FlattenNonContiguous) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32);
    auto transposed = t->Transpose(0, 1);  
    auto flattened = transposed->Flatten(0, -1);
    
    EXPECT_EQ(flattened->Dims(), std::vector<int64_t>{12});

    EXPECT_EQ(flattened->NumElements(), 12);
}
