#include "tensorrt_engine.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

namespace hs {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

void Logger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[TRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[TRT WARN] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[TRT INFO] " << msg << std::endl;
            break;
        default:
            break;
    }
}

static void delete_trt_runtime(nvinfer1::IRuntime* p) { if (p) p->destroy(); }
static void delete_trt_engine(nvinfer1::ICudaEngine* p) { if (p) p->destroy(); }
static void delete_trt_context(nvinfer1::IExecutionContext* p) { if (p) p->destroy(); }

TensorRTEngine::TensorRTEngine(const TensorRTConfig& config)
    : config_(config)
    , runtime_(nullptr, delete_trt_runtime)
    , engine_(nullptr, delete_trt_engine)
    , context_(nullptr, delete_trt_context)
{
    if (config_.batch_size <= 0) {
        config_.batch_size = 1024;
    }
    set_device();
    load_engine();
    create_context();
    allocate_buffers();
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

TensorRTEngine::~TensorRTEngine() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

void TensorRTEngine::set_device() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    if (config_.device_id >= device_count) {
        config_.device_id = 0;
    }
    CUDA_CHECK(cudaSetDevice(config_.device_id));
}

void TensorRTEngine::load_engine() {
    std::ifstream file(config_.engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open engine file: " + config_.engine_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + config_.engine_path);
    }
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size, nullptr));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    const int num_bindings = engine_->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            input_index_ = i;
        } else {
            output_index_ = i;
        }
    }
    if (input_index_ < 0 || output_index_ < 0) {
        throw std::runtime_error("Failed to find input/output bindings");
    }
}

void TensorRTEngine::create_context() {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
    nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
    input_dims.d[0] = config_.batch_size;
    context_->setBindingDimensions(input_index_, input_dims);
}

void TensorRTEngine::allocate_buffers() {
    nvinfer1::Dims input_dims = context_->getBindingDimensions(input_index_);
    nvinfer1::Dims output_dims = context_->getBindingDimensions(output_index_);

    input_size_bytes_ = config_.batch_size * config_.input_channels * sizeof(float);
    output_size_bytes_ = config_.batch_size * config_.num_classes * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_, input_size_bytes_));
    CUDA_CHECK(cudaMalloc(&d_output_, output_size_bytes_));
}

__global__ void argmax_kernel(const float* logits, int8_t* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* row = logits + idx * num_classes;
        float max_val = row[0];
        int8_t max_idx = 0;
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > max_val) {
                max_val = row[c];
                max_idx = static_cast<int8_t>(c);
            }
        }
        output[idx] = max_idx;
    }
}

void TensorRTEngine::infer_batch(const float* input, int8_t* output, int num_pixels) {
    if (!input || !output || num_pixels <= 0) return;

    const int batch = config_.batch_size;
    const int channels = config_.input_channels;
    const int classes = config_.num_classes;

    std::vector<float> h_output_logits(batch * classes);

    for (int pixel_base = 0; pixel_base < num_pixels; pixel_base += batch) {
        int current_batch = std::min(batch, num_pixels - pixel_base);
        size_t input_bytes = current_batch * channels * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(d_input_, input + pixel_base * channels,
            input_bytes, cudaMemcpyHostToDevice, stream_));

        void* bindings[2];
        bindings[input_index_] = d_input_;
        bindings[output_index_] = d_output_;

        if (current_batch < batch) {
            nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
            input_dims.d[0] = current_batch;
            context_->setBindingDimensions(input_index_, input_dims);
        }

        if (!context_->enqueueV2(bindings, stream_, nullptr)) {
            throw std::runtime_error("TensorRT enqueueV2 failed");
        }

        size_t output_bytes = current_batch * classes * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(h_output_logits.data(), d_output_,
            output_bytes, cudaMemcpyDeviceToHost, stream_));

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        for (int i = 0; i < current_batch; ++i) {
            const float* row = h_output_logits.data() + i * classes;
            float max_val = row[0];
            int8_t max_idx = 0;
            for (int c = 1; c < classes; ++c) {
                if (row[c] > max_val) {
                    max_val = row[c];
                    max_idx = static_cast<int8_t>(c);
                }
            }
            output[pixel_base + i] = max_idx;
        }

        if (current_batch < batch) {
            nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
            input_dims.d[0] = batch;
            context_->setBindingDimensions(input_index_, input_dims);
        }
    }
}

} // namespace hs
