#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace hs {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

struct TensorRTConfig {
    std::string engine_path;
    int input_channels;
    int num_classes;
    int batch_size = 1024;
    int device_id = 0;
};

class TensorRTEngine {
public:
    explicit TensorRTEngine(const TensorRTConfig& config);
    ~TensorRTEngine();

    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    void infer_batch(const float* input, int8_t* output, int num_pixels);

    int input_channels() const { return config_.input_channels; }
    int num_classes() const { return config_.num_classes; }
    int batch_size() const { return config_.batch_size; }

private:
    void load_engine();
    void create_context();
    void allocate_buffers();
    void set_device();

    TensorRTConfig config_;
    Logger logger_;

    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_;

    void* d_input_{nullptr};
    void* d_output_{nullptr};

    size_t input_size_bytes_;
    size_t output_size_bytes_;

    cudaStream_t stream_{nullptr};

    int input_index_{-1};
    int output_index_{-1};
};

} // namespace hs
