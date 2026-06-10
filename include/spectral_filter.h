#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace hs {

struct SpectralFilterConfig {
    float ndvi_threshold = 0.3f;
    float red_edge_derivative_threshold = 0.005f;
    float nir_wavelength_nm = 800.0f;
    float red_wavelength_nm = 680.0f;
    float red_edge_start_nm = 680.0f;
    float red_edge_end_nm = 750.0f;
    bool enable_ndvi = true;
    bool enable_red_edge = true;
    int8_t non_vegetation_label = 0;
};

struct SpectralFilterResult {
    std::vector<int8_t> full_mask;
    std::vector<int32_t> vegetation_indices;
    std::vector<float> ndvi_values;
    int total_pixels;
    int vegetation_pixels;
    int filtered_pixels;
    float vegetation_ratio;
    float filter_time_ms;
};

class SpectralFilter {
public:
    explicit SpectralFilter(const SpectralFilterConfig& config);

    SpectralFilterResult filter(
        const float* data,
        int num_pixels,
        int num_bands,
        const std::vector<float>& wavelengths);

    SpectralFilterResult filter(
        const float* data,
        int num_pixels,
        int num_bands,
        int nir_band_idx,
        int red_band_idx,
        int red_edge_start_idx,
        int red_edge_end_idx);

    void extract_vegetation_pixels(
        const float* data,
        int num_pixels,
        int num_bands,
        const SpectralFilterResult& filter_result,
        float* output);

    static int find_closest_band(const std::vector<float>& wavelengths, float target_nm);

    static float compute_ndvi(float nir, float red);

    static float compute_red_edge_derivative(
        const float* spectrum,
        int num_bands,
        int start_idx,
        int end_idx);

    static std::vector<float> compute_first_derivative(
        const float* spectrum,
        int num_bands,
        const std::vector<float>& wavelengths);

    const SpectralFilterConfig& config() const { return config_; }

private:
    SpectralFilterConfig config_;
};

} // namespace hs
