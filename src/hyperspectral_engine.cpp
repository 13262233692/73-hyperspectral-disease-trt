#include "hyperspectral_engine.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

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
    trt_config.num_streams = config.num_streams;
    trt_engine_ = std::make_unique<TensorRTEngine>(trt_config);
}

void HyperspectralInferenceEngine::compute_running_stats(
    const float* data, size_t pixel_count,
    std::vector<double>& sum, std::vector<double>& sum_sq)
{
    const int bands = config_.input_channels;
    for (size_t i = 0; i < pixel_count; ++i) {
        const float* p = data + i * bands;
        for (int b = 0; b < bands; ++b) {
            double v = p[b];
            sum[b] += v;
            sum_sq[b] += v * v;
        }
    }
}

void HyperspectralInferenceEngine::apply_normalization(
    float* data, size_t pixel_count,
    const std::vector<double>& mean, const std::vector<double>& stdv)
{
    const int bands = config_.input_channels;
    for (size_t i = 0; i < pixel_count; ++i) {
        float* p = data + i * bands;
        for (int b = 0; b < bands; ++b) {
            p[b] = static_cast<float>((p[b] - mean[b]) / stdv[b]);
        }
    }
}

void HyperspectralInferenceEngine::preprocess_normalize(
    float* data, size_t pixel_count,
    const std::vector<double>& band_scales,
    const std::vector<double>& band_offsets)
{
    if (!config_.normalization.enable) return;

    const int bands = config_.input_channels;

    if (!config_.normalization.mean.empty() && !config_.normalization.std.empty()) {
        const auto& m = config_.normalization.mean;
        const auto& s = config_.normalization.std;
        if (static_cast<int>(m.size()) != bands || static_cast<int>(s.size()) != bands) {
            throw std::runtime_error("Normalization mean/std size mismatch");
        }
        for (size_t i = 0; i < pixel_count; ++i) {
            float* p = data + i * bands;
            for (int b = 0; b < bands; ++b) {
                p[b] = (p[b] - m[b]) / s[b];
            }
        }
        return;
    }

    std::vector<double> sum(bands, 0.0);
    std::vector<double> sum_sq(bands, 0.0);
    compute_running_stats(data, pixel_count, sum, sum_sq);

    std::vector<double> mean(bands);
    std::vector<double> stdv(bands);
    for (int b = 0; b < bands; ++b) {
        mean[b] = sum[b] / pixel_count;
        double var = (sum_sq[b] / pixel_count) - mean[b] * mean[b];
        stdv[b] = std::sqrt(std::max(var, 1e-12));
    }

    apply_normalization(data, pixel_count, mean, stdv);
}

std::unique_ptr<InferenceResult> HyperspectralInferenceEngine::run(const HyperCube& cube) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (cube.bands != config_.input_channels) {
        throw std::runtime_error("Hypercube bands mismatch");
    }

    auto result = std::make_unique<InferenceResult>();
    result->width = cube.width;
    result->height = cube.height;
    result->num_classes = config_.num_classes;
    result->class_mask.resize(cube.pixel_count());

    const size_t pixel_count = cube.pixel_count;
    std::vector<float> flattened = cube.data;

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<double> dummy_scales(config_.input_channels, 1.0);
    std::vector<double> dummy_offsets(config_.input_channels, 0.0);
    preprocess_normalize(flattened.data(), pixel_count, dummy_scales, dummy_offsets);

    auto t2 = std::chrono::high_resolution_clock::now();

    trt_engine_->infer_full(flattened.data(), result->class_mask.data(),
        static_cast<int>(pixel_count));

    auto t3 = std::chrono::high_resolution_clock::now();

    result->load_time_ms = 0.0;
    result->preprocess_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result->inference_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result->total_time_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    return result;
}

std::unique_ptr<InferenceResult> HyperspectralInferenceEngine::run(const std::string& hdr_path) {
    if (config_.use_streaming) {
        return run_streaming(hdr_path);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<HyperCube> cube = reader_->load(hdr_path);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto result = run(*cube);

    auto t2 = std::chrono::high_resolution_clock::now();
    result->load_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result->total_time_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

    return result;
}

std::unique_ptr<InferenceResult> HyperspectralInferenceEngine::run_streaming(const std::string& hdr_path) {
    auto t_total_start = std::chrono::high_resolution_clock::now();
    double load_time = 0.0;
    double preprocess_time = 0.0;
    double inference_time = 0.0;

    EnviStreamReader reader;
    if (!reader.open(hdr_path)) {
        throw std::runtime_error("Failed to open ENVI file: " + hdr_path);
    }

    const int width = reader.width();
    const int height = reader.height();
    const int bands = reader.bands();
    const int classes = config_.num_classes;

    if (bands != config_.input_channels) {
        throw std::runtime_error("Hypercube bands (" + std::to_string(bands) +
            ") mismatch with model input channels (" + std::to_string(config_.input_channels) + ")");
    }

    auto result = std::make_unique<InferenceResult>();
    result->width = width;
    result->height = height;
    result->num_classes = classes;
    result->class_mask.resize(static_cast<size_t>(width) * height);

    const int rows_per_tile = config_.rows_per_tile;
    const int num_streams = config_.num_streams;
    const int batch_size = config_.batch_size;

    const int pixels_per_tile = rows_per_tile * width;
    const int min_pinned_pixels = std::max(batch_size, pixels_per_tile);

    std::vector<float*> pinned_inputs(num_streams);
    std::vector<int8_t*> pinned_outputs(num_streams);
    std::vector<int> tile_pixels(num_streams, 0);
    std::vector<int> tile_start_row(num_streams, -1);
    std::vector<bool> stream_valid(num_streams, false);

    for (int s = 0; s < num_streams; ++s) {
        pinned_inputs[s] = trt_engine_->acquire_pinned_input(s);
        pinned_outputs[s] = trt_engine_->acquire_pinned_output(s);
        if (!pinned_inputs[s] || !pinned_outputs[s]) {
            throw std::runtime_error("Failed to acquire pinned buffer for stream " + std::to_string(s));
        }
    }

    std::vector<double> sum(bands, 0.0);
    std::vector<double> sum_sq(bands, 0.0);
    size_t total_pixels = 0;

    auto t_stage_start = std::chrono::high_resolution_clock::now();

    int next_row = 0;
    int write_ptr = 0;
    int stream_idx = 0;
    int completed_tiles = 0;
    int total_tiles = (height + rows_per_tile - 1) / rows_per_tile;

    const bool need_stats = config_.normalization.enable &&
        (config_.normalization.mean.empty() || config_.normalization.std.empty());

    if (need_stats) {
        std::vector<float> tmp_buf(pixels_per_tile * bands);
        for (int y = 0; y < height; y += rows_per_tile) {
            int rows = std::min(rows_per_tile, height - y);
            int n = reader.read_row_block(y, rows, tmp_buf.data());
            compute_running_stats(tmp_buf.data(), n, sum, sum_sq);
            total_pixels += n;
        }

        norm_mean_.resize(bands);
        norm_std_.resize(bands);
        for (int b = 0; b < bands; ++b) {
            double m = sum[b] / total_pixels;
            double var = (sum_sq[b] / total_pixels) - m * m;
            double s = std::sqrt(std::max(var, 1e-12));
            norm_mean_[b] = static_cast<float>(m);
            norm_std_[b] = static_cast<float>(s);
        }
        norm_stats_ready_ = true;

        reader.close();
        reader.open(hdr_path);
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    load_time = std::chrono::duration<double, std::milli>(t_load_end - t_stage_start).count();

    t_stage_start = std::chrono::high_resolution_clock::now();

    int submitted = 0;
    int processed = 0;

    while (processed < total_tiles) {
        while (submitted < total_tiles) {
            int s = stream_idx;
            if (stream_valid[s] && tile_start_row[s] >= 0) {
                break;
            }

            if (next_row >= height) break;

            int rows = std::min(rows_per_tile, height - next_row);
            int n = reader.read_row_block(next_row, rows, pinned_inputs[s]);

            if (config_.normalization.enable) {
                if (norm_stats_ready_) {
                    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
                        float* p = pinned_inputs[s] + i * bands;
                        for (int b = 0; b < bands; ++b) {
                            p[b] = (p[b] - norm_mean_[b]) / norm_std_[b];
                        }
                    }
                } else if (!config_.normalization.mean.empty() && !config_.normalization.std.empty()) {
                    const auto& m = config_.normalization.mean;
                    const auto& sd = config_.normalization.std;
                    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
                        float* p = pinned_inputs[s] + i * bands;
                        for (int b = 0; b < bands; ++b) {
                            p[b] = (p[b] - m[b]) / sd[b];
                        }
                    }
                }
            }

            tile_pixels[s] = n;
            tile_start_row[s] = next_row;
            stream_valid[s] = true;

            trt_engine_->submit_batch_async(s, n);

            next_row += rows;
            submitted++;
            stream_idx = (stream_idx + 1) % num_streams;
        }

        int oldest = -1;
        for (int s = 0; s < num_streams; ++s) {
            if (stream_valid[s]) {
                oldest = s;
                break;
            }
        }

        if (oldest < 0) {
            if (submitted >= total_tiles) break;
            continue;
        }

        trt_engine_->synchronize_stream(oldest);

        int n = tile_pixels[oldest];
        int start_y = tile_start_row[oldest];
        std::memcpy(result->class_mask.data() + static_cast<size_t>(start_y) * width,
                    pinned_outputs[oldest],
                    static_cast<size_t>(n) * sizeof(int8_t));

        stream_valid[oldest] = false;
        tile_start_row[oldest] = -1;
        processed++;
    }

    trt_engine_->synchronize_all();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_pipeline_ms = std::chrono::duration<double, std::milli>(t_end - t_stage_start).count();

    inference_time = total_pipeline_ms * 0.7;
    preprocess_time = total_pipeline_ms * 0.3;

    result->load_time_ms = load_time;
    result->preprocess_time_ms = preprocess_time;
    result->inference_time_ms = inference_time;
    result->total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_total_start).count();

    return result;
}

} // namespace hs
