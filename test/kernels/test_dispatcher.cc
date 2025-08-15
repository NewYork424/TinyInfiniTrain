#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/autograd/matmul.h"

using namespace infini_train;

// 测试用 Kernel 函数
void TestKernel1(float* param) { *param += 1.0f; }
int TestKernel2(int a, int b) { return a + b; }

TEST(DispatcherTest, RegisterAndGetKernel) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );

    REGISTER_KERNEL(DeviceType::kCPU, TestKernel1, TestKernel1);
    REGISTER_KERNEL(DeviceType::kCUDA, TestKernel2, TestKernel2);

    auto kernel1 = Dispatcher::Instance().GetKernel({DeviceType::kCPU, "TestKernel1"});
    float val = 0.0f;
    kernel1.Call<void>(&val);  // 调用 Kernel
    EXPECT_FLOAT_EQ(val, 1.0f);

    auto kernel2 = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "TestKernel2"});
    EXPECT_EQ(kernel2.Call<int>(2, 3), 5);
}

TEST(DispatcherTest, DuplicateRegistration) {
    auto& dispatcher = Dispatcher::Instance();
    auto key = std::make_pair(DeviceType::kCPU, "TestKernel");
    
    REGISTER_KERNEL(DeviceType::kCPU, TestKernel, TestKernel1);
    
    EXPECT_DEATH(
        REGISTER_KERNEL(DeviceType::kCPU, TestKernel, TestKernel1),
        "Kernel already registered"
    );
}

TEST(DispatcherTest, GetNonexistentKernel) {
    auto& dispatcher = Dispatcher::Instance();
    auto key = std::make_pair(DeviceType::kCPU, "NonExistentKernel");
    
    EXPECT_DEATH(
        dispatcher.GetKernel(key),
        "Kernel not found"
    );
}
