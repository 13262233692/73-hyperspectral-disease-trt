#include "hyperspectral_engine.h"

#include <iostream>
#include <algorithm>
#include <numeric>

namespace hs {

HyperspectralInferenceEngine::HyperspectralInferenceEngine(const Config& config)
    : config_(config)
    , reader_(std::make_unique<EnviReader>())
{
    TensorRTConfig trt_config;
    trt_config.engine_path = config.engine_path;
    trt_config.input_channels = config.input_channels;
    trt_config.num_classes = config.num_classes;
    trt_config.batch_size = config.batch_size;
    trt_config.device_id = config.device_id;
    trt_engine_ = std::make_unique<TensorRTEngine>(trt_config);
}

void HyperspectralInferenceEngine::preprocess_normalize(float* data, size_t pixel_count) {
    if (!config_.normalization.enable) return;

    const int bands = config_.input_channels;

    if (config_.normalization.mean.empty() || config_.normalization.std.empty()) {
        std::vector<double> sum(bands, 0.0);
        std::vector<double> sum_sq(bands, 0.0);
        for (size_t i = 0; i < pixel_count; ++i) {
            for (int b = 0; b < bands; ++b) {
                float v = data[i * bands + b];
                sum[b] += v;
                sum_sq[b] += static_cast<double>(v) * v;
            }
        }
        for (int b = 0; b < bands; ++b) {
            double mean = sum[b] / pixel_count;
            double var = (sum_sq[b] / pixel_count) - mean * mean;
            double std = std::sqrt(std::max(var, 1e-12));
            for (size_t i = 0; i < pixel_count; ++i) {
                data[i * bands + b] = static_cast<float>((data[i * bands + b] - mean) / std);
            }
        }
    } else {
        const auto& mean = config_.normalization.mean;
        const auto& std = config_.normalization.std;
        if (static_cast<int>(mean.size()) != bands || static_cast<int>(std.size()) != bands) {
            throw std::runtime_error("Normalization mean/std size mismatch");
        }
        for (size_t i = 0; i < pixel_count; ++i) {
            for (int b = 0; b < bands; ++b) {
                data[i * bands + b] = static_cast<float>((data[i * bands + b] - mean[b]) / std[b]);
            }
        }
    }
}

std::unique_ptr<InferenceResult> HyperspectralInferenceEngine::run(const std::string& hdr_path) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::unique_ptr<HyperCube> cube = reader_->load(hdr_path);

    auto t1 = std::chrono::high_resolution_clock::now();

    return run(*cube);
}

std::unique_ptr<InferenceResult> HyperspectralInferenceEngine::run(const HyperCube& cube) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (cube.bands != config_.input_channels) {
        throw std::runtime_error("Hypercube bands (" + std::to_string(cube.bands) +
            ") mismatch with model input channels (" + std::to_string(config_.input_channels) + ")");
    }

    auto result = std::make_unique<InferenceResult>();
    result->width = cube.width;
    result->height = cube.height;
    result->num_classes = config_.num_classes;
    result->class_mask.resize(cube.pixel_count());

    const size_t pixel_count = cube.pixel_count();
    std::vector<float> flattened = cube.data;

    auto t1 = std::chrono::high_resolution_clock::now();

    preprocess_normalize(flattened.data(), pixel_count);

    auto t2 = std::chrono::high_resolution_clock::now();

    trt_engine_->infer_batch(flattened.data(), result->class_mask.data(),
        static_cast<int>(pixel_count));

    auto t3 = std::chrono::high_resolution_clock::now();

    result->load_time_ms = 0.0;
    result->preprocess_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result->inference_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result->total_time_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    return result;
}

} // namespace hs
