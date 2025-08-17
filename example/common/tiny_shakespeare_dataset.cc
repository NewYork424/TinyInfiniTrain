#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

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
    uint32_t magic = BytesToType<uint32_t>(header_bytes, 8);
    LOG(INFO) << "Parsed magic number: " << magic;

    // 解析 version (4 bytes)
    uint32_t version = BytesToType<uint32_t>(header_bytes, 0);
    LOG(INFO) << "Parsed version: " << version;
    CHECK(kTypeMap.contains(version)) << "Unsupported version: " << version;
    TinyShakespeareType type = kTypeMap.at(version);

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
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================

    text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
    num_samples_ = text_file_.dims[0] - 1;  // 减1是因为每个样本需要下一个token作为标签
    sequence_size_in_bytes_ = sequence_length * kTypeToSize.at(text_file_.type);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
