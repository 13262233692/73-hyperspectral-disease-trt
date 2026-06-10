#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include "envi_reader.h"
#include "tensorrt_engine.h"

namespace hs {

struct InferenceResult {
    int width;
    int height;
    int num_classes;
    std::vector<int8_t> class_mask;
    double load_time_ms;
    double preprocess_time_ms;
    double inference_time_ms;
    double total_time_ms;

    int8_t get_class(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return -1;
        return class_mask[static_cast<size_t>(y) * width + x];
    }
};

struct SpectralNormalization {
    bool enable = true;
    std::vector<float> mean;
    std::vector<float> std;
};

class HyperspectralInferenceEngine {
public:
    struct Config {
        std::string engine_path;
        int input_channels;
        int num_classes;
        int batch_size = 2048;
        int device_id = 0;
        SpectralNormalization normalization;
    };

    explicit HyperspectralInferenceEngine(const Config& config);
    ~HyperspectralInferenceEngine() = default;

    HyperspectralInferenceEngine(const HyperspectralInferenceEngine&) = delete;
    HyperspectralInferenceEngine& operator=(const HyperspectralInferenceEngine&) = delete;

    std::unique_ptr<InferenceResult> run(const std::string& hdr_path);

    std::unique_ptr<InferenceResult> run(const HyperCube& cube);

private:
    void preprocess_normalize(float* data, size_t pixel_count);

    Config config_;
    std::unique_ptr<EnviReader> reader_;
    std::unique_ptr<TensorRTEngine> trt_engine_;
};

} // namespace hs
