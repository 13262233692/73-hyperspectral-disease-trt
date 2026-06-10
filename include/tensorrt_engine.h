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
    int batch_size = 4096;
    int device_id = 0;
    int num_streams = 2;
};

struct PinnedBufferPair {
    float* h_input{nullptr};
    float* d_input{nullptr};
    float* d_output_logits{nullptr};
    int8_t* d_output_labels{nullptr};
    int8_t* h_output_labels{nullptr};
    cudaStream_t stream{nullptr};
    cudaEvent_t ready_event{nullptr};
    size_t input_bytes{0};
    size_t output_logits_bytes{0};
    size_t output_labels_bytes{0};
    int current_batch{0};
    bool in_flight{false};
};

class TensorRTEngine {
public:
    explicit TensorRTEngine(const TensorRTConfig& config);
    ~TensorRTEngine();

    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    void infer_full(const float* input, int8_t* output, int num_pixels);

    int input_channels() const { return config_.input_channels; }
    int num_classes() const { return config_.num_classes; }
    int batch_size() const { return config_.batch_size; }
    int num_streams() const { return config_.num_streams; }

    float* acquire_pinned_input(int stream_idx);
    int8_t* acquire_pinned_output(int stream_idx);
    void submit_batch_async(int stream_idx, int batch_size);
    void synchronize_stream(int stream_idx);
    void synchronize_all();

    size_t pinned_input_bytes(int stream_idx) const;
    size_t pinned_output_bytes(int stream_idx) const;

private:
    void load_engine();
    void create_context();
    void allocate_pinned_buffers();
    void set_device();
    void launch_argmax(int stream_idx, int batch_size);

    TensorRTConfig config_;
    Logger logger_;

    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>> contexts_;

    std::vector<PinnedBufferPair> buffers_;

    int input_index_{-1};
    int output_index_{-1};
};

} // namespace hs
