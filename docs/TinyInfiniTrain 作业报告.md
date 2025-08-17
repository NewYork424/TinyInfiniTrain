# TinyInfiniTrain 作业报告

## 一、test 通过截图

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

本次任务的核心是理解并运用框架的`Dispatcher`机制，它作为硬件抽象层，负责解耦通用计算逻辑与具体设备（CPU/CUDA）的实现。

1.  前向传播 (Forward):
    *   首先，从输入张量`input`中获取其所在的设备类型（如`kCPU`或`kCUDA`）。
    *   然后，构造一个`Key`（包含设备类型和操作名称"NegForward"），并调用`Dispatcher::Instance().GetKernel()`来获取一个封装了设备专属函数的`KernelFunction`对象。
    *   最后，通过`kernel.Call()`方法，以类型安全的方式调用该函数，并传入输入张量，最终返回计算结果。

2.  反向传播 (Backward):
    *   思路与前向传播类似。根据链式法则，`Neg`操作的梯度等于上游传来的梯度乘以-1。
    *   我们从上游梯度`grad_output`中获取设备信息，并向`Dispatcher`请求"NegBackward"的`kernel`。
    *   调用该`kernel`处理`grad_output`，计算并返回当前操作对输入的梯度。

#### 遇到问题

此部分的设计遵循了框架的指导思想，接口清晰明了。得益于`Dispatcher`的良好封装，实现过程非常顺利，没有遇到显著困难。


### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
        const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims.back();
    const int64_t N = other_dims.back();

    const int64_t num_batches = input->NumElements() / (M * K);
    const int64_t input_matrix_size = M * K;
    const int64_t other_matrix_size = K * N;
    const int64_t output_matrix_size = M * N;

    auto *input_data = static_cast<float *>(input->DataPtr());
    auto *other_data = static_cast<float *>(other->DataPtr());
    auto *output_data = static_cast<float *>(output->DataPtr());

    for (int64_t i = 0; i < num_batches; ++i) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_mat(
            input_data + i * input_matrix_size, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> other_mat(
            other_data + i * other_matrix_size, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_mat(
            output_data + i * output_matrix_size, M, N);
        output_mat = input_mat * other_mat;
    }

    return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
        auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32, other->GetDevice());

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims.back();
    const int64_t N = other_dims.back();

    const int64_t num_batches = input->NumElements() / (M * K);
    const int64_t input_matrix_size = M * K;
    const int64_t other_matrix_size = K * N;
    const int64_t output_matrix_size = M * N;

    auto *input_data = static_cast<float *>(input->DataPtr());
    auto *other_data = static_cast<float *>(other->DataPtr());
    auto *grad_output_data = static_cast<float *>(grad_output->DataPtr());
    auto *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_data = static_cast<float *>(grad_other->DataPtr());

    for (int64_t i = 0; i < num_batches; ++i) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_mat(
            input_data + i * input_matrix_size, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> other_mat(
            other_data + i * other_matrix_size, K, N);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_output_mat(
            grad_output_data + i * output_matrix_size, M, N);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_input_mat(
            grad_input_data + i * input_matrix_size, M, K);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_other_mat(
            grad_other_data + i * other_matrix_size, K, N);

        grad_input_mat = grad_output_mat * other_mat.transpose();
        grad_other_mat = input_mat.transpose() * grad_output_mat;
    }

    return {grad_input, grad_other};
    }
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]); // K 对齐

    // 计算 batch 数：逐 batch 做 [M,K] x [K,N]
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims.back();
    const int64_t N = other_dims.back();

    const int64_t num_batches = input->NumElements() / (M * K);
    CHECK_EQ(other->NumElements(), num_batches * K * N); // 保证 batch 对齐

    auto output_dims = input_dims;
    output_dims.back() = N;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 利用 row-major -> column-major 转换：
    // C_row (M,N) -> C_col (N,M) = B_row^T (N,K) * A_row^T (K,M)
    // 于是调用: m=N, n=M, k=K, A=other(视作 N x K), B=input(视作 K x M)
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

    const int64_t strideA = K * N; // other 每个 batch 大小 (K,N) row-major == (N,K) col-major
    const int64_t strideB = M * K; // input 每个 batch
    const int64_t strideC = M * N; // output 每个 batch (row) == (N,M) col

    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           (int)N, (int)M, (int)K,
                                           &alpha,
                                           static_cast<const float*>(other->DataPtr()), (int)N, strideA,
                                           static_cast<const float*>(input->DataPtr()), (int)K, strideB,
                                           &beta,
                                           static_cast<float*>(output->DataPtr()), (int)N, strideC,
                                           (int)num_batches));

    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
            const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_GE(grad_dims.size(), 2);

    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims.back();
    const int64_t N = other_dims.back();
    CHECK_EQ(grad_dims[grad_dims.size() - 2], M);
    CHECK_EQ(grad_dims.back(), N);
    CHECK_EQ(other_dims[other_dims.size() - 2], K);

    const int64_t num_batches = input->NumElements() / (M * K);
    CHECK_EQ(other->NumElements(), num_batches * K * N);
    CHECK_EQ(grad_output->NumElements(), num_batches * M * N);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

    // 1) grad_input_row (M,K): compute grad_input_row^T (K,M) = other_row * grad_output_row^T
    // 使用: C = op(A)*op(B)
    // A: other buffer 视作 (N,K) col-major, op(A)=T -> (K,N)
    // B: grad_output buffer 视作 (N,M) col-major, op(B)=N -> (N,M)
    // C: grad_input^T (K,M)
    const int64_t strideOther = K * N;
    const int64_t strideGradOut = M * N;
    const int64_t strideGradIn = M * K;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           (int)K, (int)M, (int)N,
                                           &alpha,
                                           static_cast<const float*>(other->DataPtr()), (int)N, strideOther,
                                           static_cast<const float*>(grad_output->DataPtr()), (int)N, strideGradOut,
                                           &beta0,
                                           static_cast<float*>(grad_input->DataPtr()), (int)K, strideGradIn,
                                           (int)num_batches));

    // 2) grad_other_row (K,N): compute grad_other_row^T (N,K) = grad_output_row^T (N,M) * input_row (M,K)
    const int64_t strideInput = M * K;
    const int64_t strideGradOther = K * N;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_T,
                                           (int)N, (int)K, (int)M,
                                           &alpha,
                                           static_cast<const float*>(grad_output->DataPtr()), (int)N, strideGradOut,
                                           static_cast<const float*>(input->DataPtr()), (int)K, strideInput,
                                           &beta0,
                                           static_cast<float*>(grad_other->DataPtr()), (int)N, strideGradOther,
                                           (int)num_batches));

    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_other};
    }
```

#### 解决思路

CUDA上的实现主要依赖于NVIDIA的cuBLAS库，它提供了高度优化的BLAS（基础线性代数子程序）实现。

1.  前向计算:
    *   我们选用`cublasSgemmStridedBatched`函数，它专门用于执行批处理的矩阵乘法，性能极高。
    *   一个关键的挑战是内存布局：`Tensor`（和C++）默认是行主序（Row-Major），而cuBLAS（和Fortran）默认是列主序（Column-Major）。直接调用会导致结果错误。
    *   解决方案是利用数学恒等式 `(A * B)^T = B^T * A^T`。我们将行主序的`C = A * B`运算，转换为等价的列主序运算 `C^T = B^T * A^T`。在调用`cublasSgemmStridedBatched`时，我们交换输入矩阵的顺序，并调整M、N、K以及转置参数，从而在不进行任何内存重排的情况下得到正确的结果。

2.  反向传播:
    *   梯度的计算同样转换为`cublasSgemmStridedBatched`调用。
    *   `grad_input = grad_output * other^T` 对应一次批处理乘法。
    *   `grad_other = input^T * grad_output` 对应另一次批处理乘法。
    *   与前向计算类似，我们为每次调用精心设置操作标志（`CUBLAS_OP_N`或`CUBLAS_OP_T`）来处理矩阵转置，以适应cuBLAS的列主序要求。

#### 遇到问题
主要挑战在于正确处理高维（批处理）张量。单纯的2D矩阵乘法逻辑无法直接扩展。关键在于深刻理解高维张量在连续内存上的布局方式，并精确计算出`cublasSgemmStridedBatched`所需的步长（stride）参数。这个参数告诉cuBLAS每个批次的矩阵数据在内存中的间隔距离。最初在步长计算上出现偏差，导致计算结果混乱，通过参考cuBLAS文档和逐步调试得以解决。


### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
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
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
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

    // 配置 CUDA 内核
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // 启动 CUDA 内核
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate, beta1,
                                                                beta2, eps, beta1_t, beta2_t,
                                                                num_elements );
}
```

#### 解决思路

实现遵循Adam优化器标准算法。该算法结合了一阶动量（Momentum）和二阶动量（RMSProp）的优点。

1.  核心更新逻辑:
    *   更新一阶动量 `m`: `m = beta1 * m + (1 - beta1) * grad`
    *   更新二阶动量 `v`: `v = beta2 * v + (1 - beta2) * grad^2`
    *   计算偏差修正: 由于`m`和`v`在初期会被初始化为0，它们会偏向于0。因此需要进行偏差修正，计算修正后的学习率 `lr_t`。
    *   更新参数: `param -= lr_t * m / (sqrt(v) + epsilon)`

2.  CPU实现: 直接在一个循环中遍历所有参数元素，并应用上述更新规则。

3.  CUDA实现: 将上述更新逻辑封装在一个`__global__`函数（CUDA Kernel）中。每个CUDA线程负责一个或多个参数的更新。通过标准的网格-线程块（Grid-Block）配置，让GPU并行地对所有参数执行更新操作，从而实现大规模加速。

#### 遇到问题

CUDA版本的实现初期遇到了运行时错误，但错误信息并非来自`Adam`核函数本身。经过排查，问题根源在于执行环境的配置。我们的测试框架依赖`srun`（或类似工具）来分配和初始化GPU资源。在未指定GPU设备的情况下直接运行测试，导致CUDA上下文未能正确建立，任何CUDA API（包括cuBLAS和自定义核函数）的调用都会失败。这提醒我们，GPU编程的正确性不仅依赖于代码逻辑，同样依赖于一个正确配置的运行时环境。


### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // =================================== 作业 ===================================
    // TODO：实现张量扁平化操作，将指定维度范围[start, end]内的所有维度合并为一个维度
    // HINT: 
    // =================================== 作业 ===================================
    if (start < 0) start += dims_.size();
    if (end < 0) end += dims_.size();
    CHECK_GE(start, 0);
    CHECK_LT(end, dims_.size());
    CHECK_LE(start, end);

    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < start; ++i) {
        new_shape.push_back(dims_[i]);
    }

    int64_t flattened_dim = 1;
    for (int64_t i = start; i <= end; ++i) {
        flattened_dim *= dims_[i];
    }
    new_shape.push_back(flattened_dim);

    for (int64_t i = end + 1; i < dims_.size(); ++i) {
        new_shape.push_back(dims_[i]);
    }

    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // HINT: 
    // =================================== 作业 ===================================
    CHECK(requires_grad_) << "Cannot call backward on a tensor that does not require grad.";
    CHECK(grad_fn_ != nullptr) << "Cannot call backward on a leaf tensor.";

    // 初始梯度：如果未提供，则创建一个全为1的梯度张量
    if (gradient == nullptr) {
        CHECK_EQ(num_elements_, 1) << "grad can be implicitly created only for scalar outputs";
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill(1.0f);
    }
    CHECK(gradient->Dims() == dims_) << "Gradient shape must match tensor shape";

    // 启动反向传播过程
    // 将初始梯度传递给当前张量的 grad_fn，并指定是哪个输出索引
    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路

`Flatten`操作旨在保持总元素数量不变的前提下，改变张量的形状。其逻辑如下：
1.  解析`start`和`end`维度索引，处理负数索引的情况。
2.  创建一个新的维度向量`new_shape`。
3.  将`start`之前的所有维度直接复制到`new_shape`中。
4.  计算`start`到`end`（包含两者）之间所有维度的乘积，得到`flattened_dim`，并将其添加到`new_shape`中。
5.  将`end`之后的所有维度复制到`new_shape`中。
6.  最后，调用`View()`方法，用`new_shape`创建一个共享底层数据但具有新形状的`Tensor`。为确保`View()`能成功，我们先调用`Contiguous()`来保证数据在内存中是连续的。

`Backward`是自动微分的核心。它触发了从当前`Tensor`开始，沿着计算图向叶子节点回溯的梯度计算过程。
1.  梯度初始化: 反向传播需要一个初始梯度。如果调用者没有提供（通常对于标量损失函数），我们会创建一个值为1.0的梯度张量。这是链式法则的起点。
2.  合法性检查: 确保当前`Tensor`是需要梯度（`requires_grad_`）且不是叶子节点（`grad_fn_`不为空）。
3.  调用入口: 核心操作是调用当前`Tensor`的`grad_fn_->BackwardPartial()`方法。`grad_fn_`保存了创建此`Tensor`的操作（如`Add`、`Mul`等）。我们将初始梯度和当前`Tensor`在其创建者（`grad_fn_`）的输出列表中的索引`output_idx_`传递给它。
4.  递归传播: `BackwardPartial`函数会负责计算梯度，并递归地调用其输入`Tensor`的`Backward`方法，从而将梯度一步步地传播回计算图的叶子节点。

#### 遇到问题

这两个基础功能的实现较为直接，主要依赖于对`Tensor`数据结构和计算图概念的清晰理解。框架设计良好，实现过程顺畅。


### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业 ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // HINT: 
    // =================================== 作业 ===================================
    using FuncT = RetT (*)(ArgsT...);
        auto func = reinterpret_cast<FuncT>(func_ptr_);
        return func(std::forward<ArgsT>(args)...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业 ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业 ===================================
    CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static bool register_##kernel_name##_##__LINE__ = []() {                                                            \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                          \
        return true;                                                                                                   \
    }();
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
```

#### 解决思路

为了实现一个灵活且可扩展的计算框架，我们需要一套自动化的算子（Kernel）注册和调用机制。

1.  Kernel注册 (`Register`):
    *   我们使用一个全局的`Dispatcher`单例，其内部维护一个从`Key`到`KernelFunction`的哈希表`key_to_kernel_map_`。
    *   `Key`是一个`std::pair`，包含设备类型和算子名称字符串。`KernelFunction`是一个可以存储任意函数指针的类型擦除包装器。
    *   `Register`函数接收一个`Key`和一个函数指针，将它们存入哈希表中。同时进行重复注册检查，确保每个`Key`只被注册一次。

2.  自动注册宏 (`REGISTER_KERNEL`):
    *   为了避免在代码中手动调用`Register`，我们设计了一个宏。这个宏利用了C++静态变量在程序启动时初始化的特性。
    *   `REGISTER_KERNEL(device, name, func)`定义一个静态`bool`变量，调用`Dispatcher::Instance().Register()`。这样，只要包含了定义该`kernel`的文件，它就会在`main`函数执行前被自动注册到`Dispatcher`中。
    *   `##`是C++预处理器中的Token-Pasting Operator，用于将`register_`、`kernel_name`和`__LINE__`拼接成一个唯一的变量名，防止宏在同一文件中多次使用时发生命名冲突。

3.  通过`reinterpret_cast`，将其转换回调用者指定的、具有正确签名（返回类型和参数类型）的函数指针。


#### 遇到问题

主要的挑战在于设计`REGISTER_KERNEL`宏。如何确保宏在全局作用域内安全、自动地执行注册逻辑，并且不会因多次使用而产生命名冲突，是思考的重点。最终通过结合`static`变量初始化和`##`拼接唯一变量名的方式，优雅地解决了这个问题。


### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "Cannot open file: " << path;

    // 读取 header (1024 bytes)
    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &file);

    // 解析 magic number (4 bytes)
    uint32_t magic = BytesToType<uint32_t>(header_bytes, 0);
    LOG(INFO) << "Parsed magic number: " << magic;
    CHECK(kTypeMap.contains(magic)) << "Unsupported magic: " << magic;
    TinyShakespeareType type = kTypeMap.at(magic);

    // 解析 version (4 bytes)
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    LOG(INFO) << "Parsed version: " << version;
    

    // 解析 number of tokens (4 bytes)
    uint32_t num_tokens = BytesToType<uint32_t>(header_bytes, 4);
    LOG(INFO) << "Parsed number of tokens: " << num_tokens;
    // 计算样本数量
    size_t num_samples = num_tokens - sequence_length;
    CHECK_GT(num_samples, 0) << "Not enough tokens for the given sequence length";

    // 读取 token 数据
    size_t token_size = kTypeToSize.at(type);
    size_t data_size = num_tokens * token_size;
    auto data_bytes = ReadSeveralBytesFromIfstream(data_size, &file);

    // 创建 tensor
    std::vector<int64_t> dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};
    DataType data_type = kTypeToDataType.at(type);
    auto tensor = std::make_shared<infini_train::Tensor>(dims, data_type, infini_train::Device(infini_train::DeviceType::kCPU, 0));

    // 复制数据到 tensor
    std::memcpy(tensor->DataPtr(), data_bytes.data(), data_size);

    return TinyShakespeareFile{
        .type = type,
        .dims = dims,
        .tensor = *tensor
    };
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
    num_samples_ = text_file_.dims[0] - 1;  // 减1是因为每个样本需要下一个token作为标签
    sequence_size_in_bytes_ = sequence_length * kTypeToSize.at(text_file_.type);
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    // Read header
    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header_bytes, 0);
    // uint32_t version = BytesToType<uint32_t>(header_bytes, 4); // version is not a member
    vocab_size_ = BytesToType<uint32_t>(header_bytes, 8);

    // Set EOT token
    CHECK(kEotMap.count(magic_number_)) << "Unknown magic number: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    // Read vocab table
    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint32_t len;
        ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string token(len, '\0');
        ifs.read(&token[0], len);
        token_table_[i] = token;
    }
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    if (token_id < vocab_size_) {
        return token_table_[token_id];
    }
    return "[UNKNOWN]";
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* ...原代码... */
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // 前向推理获取 logits
        auto output = model.Forward({x});
        auto logits = output[0];

        // 获取最后一个位置的 logits
        auto last_logits = logits->Slice(1, sequence_length - 1, sequence_length);

        // 应用 softmax 得到概率分布
        auto probs = infini_train::nn::function::Softmax(last_logits, 2);

        // 随机采样
        float *probs_ptr = static_cast<float *>(probs->DataPtr());
        float coin = RandomF32(kRngState);
        int next_token = SampleMult(probs_ptr, vocab_size_, coin);

        // 更新输入序列（向左移动一位，添加新生成的token）
        int64_t *x_ptr = static_cast<int64_t *>(x->DataPtr());
        for (int i = 0; i < sequence_length - 1; ++i) {
            x_ptr[i] = x_ptr[i + 1];
        }
        x_ptr[sequence_length - 1] = next_token;

        // 解码并输出
        std::cout << Decode(next_token) << std::flush;
    }
    std::cout << std::endl;
}
```

#### 解决思路

此项任务是前面所有工作的综合应用，旨在打通整个训练流程，并验证其正确性。

1.  数据读取 (`ReadTinyShakespeareFile`):
    *   严格按照二进制文件格式说明进行解析。使用`std::ifstream`以二进制模式打开文件。
    *   首先读取1024字节的`HEADER`。
    *   使用辅助函数`BytesToType`从`header`字节数组的指定偏移量处解析出`magic`、`version`和`num_toks`等元数据。
    *   根据`magic` number从`kTypeMap`中确定`Token`的数据类型（如`UINT16`），并据此计算出每个`token`占用的字节数。
    *   读取文件剩余的`DATA`部分，即所有`token`数据。
    *   最后，将加载的数据和元信息封装成一个`TinyShakespeareFile`结构体，其中`Tensor`对象存储了完整的`token`序列。

2.  Tokenizer功能实现:
    *   构造函数: 同样地，根据二进制格式说明加载`tokenizer`文件，解析出`magic`、`vocab_size_`，并循环读取词表中的每一个`token`（先读长度，再读字符串）。
    *   `Decode`: 实现一个简单的查找操作，根据`token_id`在`token_table_`中返回对应的字符串。
    *   `GenerateText`: 这是文本生成的推理循环。
        *   前向传播: 将当前的输入序列`x`送入`model.Forward()`，得到`logits`。
        *   采样: 只关心序列最后一个时间步的`logits`。对其应用`Softmax`函数将其转换为概率分布，然后通过随机采样（`SampleMult`）从该分布中选择一个`next_token`。
        *   更新输入: 将输入序列`x`向左滚动一位，并在末尾填入新生成的`next_token`，作为下一次推理的输入。
        *   解码输出: 调用`Decode`将`next_token`转换为文本并打印。
        *   重复此过程直到达到指定的生成长度。

#### 遇到问题

在实现数据加载部分时，对二进制文件格式的理解是关键。最初对`header`中各个字段的偏移量（offset）和字节序（endianness）存在一些困惑。通过编写小段的调试代码，打印出从文件中读取的字节，并与Python参考实现中的打包逻辑进行比对，最终明确了正确的解析方式。确保数据类型（如`uint32_t`）和其在文件中的字节表示能够正确转换，是保证数据加载无误的核心。

