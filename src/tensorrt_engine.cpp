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

__global__ void argmax_kernel(const float* logits, int8_t* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* row = logits + static_cast<size_t>(idx) * num_classes;
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

TensorRTEngine::TensorRTEngine(const TensorRTConfig& config)
    : config_(config)
    , runtime_(nullptr, delete_trt_runtime)
    , engine_(nullptr, delete_trt_engine)
{
    if (config_.batch_size <= 0) config_.batch_size = 4096;
    if (config_.num_streams <= 0) config_.num_streams = 2;
    if (config_.num_streams > 8) config_.num_streams = 8;

    set_device();
    load_engine();
    create_context();
    allocate_pinned_buffers();
}

TensorRTEngine::~TensorRTEngine() {
    for (auto& buf : buffers_) {
        if (buf.stream) {
            cudaStreamDestroy(buf.stream);
            buf.stream = nullptr;
        }
        if (buf.ready_event) {
            cudaEventDestroy(buf.ready_event);
            buf.ready_event = nullptr;
        }
        if (buf.h_input) {
            cudaFreeHost(buf.h_input);
            buf.h_input = nullptr;
        }
        if (buf.h_output_labels) {
            cudaFreeHost(buf.h_output_labels);
            buf.h_output_labels = nullptr;
        }
        if (buf.d_input) {
            cudaFree(buf.d_input);
            buf.d_input = nullptr;
        }
        if (buf.d_output_logits) {
            cudaFree(buf.d_output_logits);
            buf.d_output_logits = nullptr;
        }
        if (buf.d_output_labels) {
            cudaFree(buf.d_output_labels);
            buf.d_output_labels = nullptr;
        }
    }
    buffers_.clear();
    contexts_.clear();
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
    contexts_.clear();
    for (int s = 0; s < config_.num_streams; ++s) {
        nvinfer1::IExecutionContext* ctx = engine_->createExecutionContext();
        if (!ctx) {
            throw std::runtime_error("Failed to create execution context " + std::to_string(s));
        }
        nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
        input_dims.d[0] = config_.batch_size;
        ctx->setBindingDimensions(input_index_, input_dims);
        contexts_.emplace_back(ctx, delete_trt_context);
    }
}

void TensorRTEngine::allocate_pinned_buffers() {
    buffers_.resize(config_.num_streams);

    const size_t input_bytes = static_cast<size_t>(config_.batch_size) * config_.input_channels * sizeof(float);
    const size_t output_logits_bytes = static_cast<size_t>(config_.batch_size) * config_.num_classes * sizeof(float);
    const size_t output_labels_bytes = static_cast<size_t>(config_.batch_size) * sizeof(int8_t);

    for (int s = 0; s < config_.num_streams; ++s) {
        auto& buf = buffers_[s];
        buf.input_bytes = input_bytes;
        buf.output_logits_bytes = output_logits_bytes;
        buf.output_labels_bytes = output_labels_bytes;
        buf.in_flight = false;
        buf.current_batch = 0;

        CUDA_CHECK(cudaHostAlloc(&buf.h_input, input_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&buf.h_output_labels, output_labels_bytes, cudaHostAllocDefault));

        CUDA_CHECK(cudaMalloc(&buf.d_input, input_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_output_logits, output_logits_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_output_labels, output_labels_bytes));

        CUDA_CHECK(cudaStreamCreateWithFlags(&buf.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&buf.ready_event, cudaEventBlockingSync));
    }
}

float* TensorRTEngine::acquire_pinned_input(int stream_idx) {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return nullptr;
    return buffers_[stream_idx].h_input;
}

int8_t* TensorRTEngine::acquire_pinned_output(int stream_idx) {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return nullptr;
    return buffers_[stream_idx].h_output_labels;
}

size_t TensorRTEngine::pinned_input_bytes(int stream_idx) const {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return 0;
    return buffers_[stream_idx].input_bytes;
}

size_t TensorRTEngine::pinned_output_bytes(int stream_idx) const {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return 0;
    return buffers_[stream_idx].output_labels_bytes;
}

void TensorRTEngine::launch_argmax(int stream_idx, int batch_size) {
    auto& buf = buffers_[stream_idx];
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    argmax_kernel<<<grid_size, block_size, 0, buf.stream>>>(
        buf.d_output_logits, buf.d_output_labels, batch_size, config_.num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void TensorRTEngine::submit_batch_async(int stream_idx, int batch_size) {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return;
    if (batch_size <= 0) return;

    auto& buf = buffers_[stream_idx];
    auto& ctx = contexts_[stream_idx];
    buf.current_batch = batch_size;

    const size_t input_bytes = static_cast<size_t>(batch_size) * config_.input_channels * sizeof(float);
    const size_t output_labels_bytes = static_cast<size_t>(batch_size) * sizeof(int8_t);

    CUDA_CHECK(cudaMemcpyAsync(buf.d_input, buf.h_input, input_bytes,
        cudaMemcpyHostToDevice, buf.stream));

    void* bindings[2];
    bindings[input_index_] = buf.d_input;
    bindings[output_index_] = buf.d_output_logits;

    if (batch_size < config_.batch_size) {
        nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
        input_dims.d[0] = batch_size;
        ctx->setBindingDimensions(input_index_, input_dims);
    }

    if (!ctx->enqueueV2(bindings, buf.stream, nullptr)) {
        throw std::runtime_error("TensorRT enqueueV2 failed on stream " + std::to_string(stream_idx));
    }

    launch_argmax(stream_idx, batch_size);

    CUDA_CHECK(cudaMemcpyAsync(buf.h_output_labels, buf.d_output_labels, output_labels_bytes,
        cudaMemcpyDeviceToHost, buf.stream));

    CUDA_CHECK(cudaEventRecord(buf.ready_event, buf.stream));
    buf.in_flight = true;

    if (batch_size < config_.batch_size) {
        nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
        input_dims.d[0] = config_.batch_size;
        ctx->setBindingDimensions(input_index_, input_dims);
    }
}

void TensorRTEngine::synchronize_stream(int stream_idx) {
    if (stream_idx < 0 || stream_idx >= config_.num_streams) return;
    auto& buf = buffers_[stream_idx];
    if (!buf.in_flight) return;
    CUDA_CHECK(cudaEventSynchronize(buf.ready_event));
    buf.in_flight = false;
}

void TensorRTEngine::synchronize_all() {
    for (int s = 0; s < config_.num_streams; ++s) {
        synchronize_stream(s);
    }
}

void TensorRTEngine::infer_full(const float* input, int8_t* output, int num_pixels) {
    if (!input || !output || num_pixels <= 0) return;

    const int batch = config_.batch_size;
    const int channels = config_.input_channels;
    const int num_streams = config_.num_streams;

    int submitted = 0;
    int completed = 0;
    int stream_ptr = 0;

    while (completed < num_pixels) {
        while (submitted < num_pixels) {
            if (buffers_[stream_ptr].in_flight) {
                break;
            }

            int cur = std::min(batch, num_pixels - submitted);
            auto& buf = buffers_[stream_ptr];

            std::memcpy(buf.h_input, input + static_cast<size_t>(submitted) * channels,
                static_cast<size_t>(cur) * channels * sizeof(float));

            submit_batch_async(stream_ptr, cur);
            submitted += cur;
            stream_ptr = (stream_ptr + 1) % num_streams;
        }

        int oldest = -1;
        for (int s = 0; s < num_streams; ++s) {
            if (buffers_[s].in_flight) {
                oldest = s;
                break;
            }
        }
        if (oldest < 0 && submitted >= num_pixels) break;
        if (oldest < 0) continue;

        synchronize_stream(oldest);

        auto& buf = buffers_[oldest];
        int actual = buf.current_batch;
        std::memcpy(output + static_cast<size_t>(completed), buf.h_output_labels,
            static_cast<size_t>(actual) * sizeof(int8_t));
        completed += actual;
    }
}

} // namespace hs
