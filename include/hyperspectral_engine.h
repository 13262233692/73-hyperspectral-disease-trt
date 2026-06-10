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
        int batch_size = 4096;
        int device_id = 0;
        int num_streams = 2;
        int rows_per_tile = 128;
        SpectralNormalization normalization;
        bool use_streaming = true;
    };

    explicit HyperspectralInferenceEngine(const Config& config);
    ~HyperspectralInferenceEngine() = default;

    HyperspectralInferenceEngine(const HyperspectralInferenceEngine&) = delete;
    HyperspectralInferenceEngine& operator=(const HyperspectralInferenceEngine&) = delete;

    std::unique_ptr<InferenceResult> run(const std::string& hdr_path);

    std::unique_ptr<InferenceResult> run(const HyperCube& cube);

    std::unique_ptr<InferenceResult> run_streaming(const std::string& hdr_path);

private:
    void preprocess_normalize(float* data, size_t pixel_count,
                              const std::vector<double>& band_scales,
                              const std::vector<double>& band_offsets);

    void compute_running_stats(const float* data, size_t pixel_count,
                               std::vector<double>& sum, std::vector<double>& sum_sq);

    void apply_normalization(float* data, size_t pixel_count,
                             const std::vector<double>& mean, const std::vector<double>& stdv);

    Config config_;
    std::unique_ptr<EnviReader> reader_;
    std::unique_ptr<TensorRTEngine> trt_engine_;

    std::vector<float> norm_mean_;
    std::vector<float> norm_std_;
    bool norm_stats_ready_{false};
};

} // namespace hs
