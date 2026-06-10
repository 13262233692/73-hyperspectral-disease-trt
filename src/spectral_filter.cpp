#include "spectral_filter.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace hs {

SpectralFilter::SpectralFilter(const SpectralFilterConfig& config)
    : config_(config)
{}

int SpectralFilter::find_closest_band(const std::vector<float>& wavelengths, float target_nm) {
    if (wavelengths.empty()) return -1;
    int best = 0;
    float best_diff = std::abs(wavelengths[0] - target_nm);
    for (int i = 1; i < static_cast<int>(wavelengths.size()); ++i) {
        float diff = std::abs(wavelengths[i] - target_nm);
        if (diff < best_diff) {
            best_diff = diff;
            best = i;
        }
    }
    return best;
}

float SpectralFilter::compute_ndvi(float nir, float red) {
    float denom = nir + red;
    if (denom < 1e-10f) return 0.0f;
    return (nir - red) / denom;
}

float SpectralFilter::compute_red_edge_derivative(
    const float* spectrum,
    int num_bands,
    int start_idx,
    int end_idx)
{
    if (start_idx < 0 || end_idx >= num_bands || start_idx >= end_idx) return 0.0f;

    float max_deriv = 0.0f;
    for (int i = start_idx + 1; i <= end_idx; ++i) {
        float deriv = spectrum[i] - spectrum[i - 1];
        if (deriv > max_deriv) {
            max_deriv = deriv;
        }
    }
    return max_deriv;
}

std::vector<float> SpectralFilter::compute_first_derivative(
    const float* spectrum,
    int num_bands,
    const std::vector<float>& wavelengths)
{
    std::vector<float> deriv(num_bands, 0.0f);
    if (num_bands < 3) return deriv;

    for (int i = 1; i < num_bands - 1; ++i) {
        float dl = wavelengths[i] - wavelengths[i - 1];
        float dr = wavelengths[i + 1] - wavelengths[i];
        if (dl < 1e-6f || dr < 1e-6f) {
            deriv[i] = 0.0f;
            continue;
        }
        deriv[i] = (spectrum[i + 1] - spectrum[i - 1]) / (dl + dr);
    }
    deriv[0] = deriv[1];
    deriv[num_bands - 1] = deriv[num_bands - 2];
    return deriv;
}

SpectralFilterResult SpectralFilter::filter(
    const float* data,
    int num_pixels,
    int num_bands,
    const std::vector<float>& wavelengths)
{
    int nir_idx = find_closest_band(wavelengths, config_.nir_wavelength_nm);
    int red_idx = find_closest_band(wavelengths, config_.red_wavelength_nm);
    int re_start_idx = find_closest_band(wavelengths, config_.red_edge_start_nm);
    int re_end_idx = find_closest_band(wavelengths, config_.red_edge_end_nm);

    if (nir_idx < 0 || red_idx < 0) {
        throw std::runtime_error("Cannot locate NIR/Red bands in wavelength data");
    }

    std::cout << "  [SpectralFilter] NIR band index: " << nir_idx
              << " (" << wavelengths[nir_idx] << " nm)" << std::endl;
    std::cout << "  [SpectralFilter] Red band index: " << red_idx
              << " (" << wavelengths[red_idx] << " nm)" << std::endl;
    if (re_start_idx >= 0 && re_end_idx >= 0) {
        std::cout << "  [SpectralFilter] Red Edge range: ["
                  << wavelengths[re_start_idx] << " nm (idx " << re_start_idx << ") - "
                  << wavelengths[re_end_idx] << " nm (idx " << re_end_idx << ")]" << std::endl;
    }

    return filter(data, num_pixels, num_bands, nir_idx, red_idx, re_start_idx, re_end_idx);
}

SpectralFilterResult SpectralFilter::filter(
    const float* data,
    int num_pixels,
    int num_bands,
    int nir_band_idx,
    int red_band_idx,
    int red_edge_start_idx,
    int red_edge_end_idx)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    SpectralFilterResult result;
    result.total_pixels = num_pixels;
    result.full_mask.resize(num_pixels, config_.non_vegetation_label);
    result.ndvi_values.resize(num_pixels, 0.0f);
    result.vegetation_pixels = 0;
    result.filtered_pixels = 0;

    for (int i = 0; i < num_pixels; ++i) {
        const float* spectrum = data + static_cast<size_t>(i) * num_bands;

        float nir = spectrum[nir_band_idx];
        float red = spectrum[red_band_idx];

        if (nir < 0.0f) nir = 0.0f;
        if (red < 0.0f) red = 0.0f;

        float ndvi = compute_ndvi(nir, red);
        result.ndvi_values[i] = ndvi;

        bool is_vegetation = true;

        if (config_.enable_ndvi && ndvi < config_.ndvi_threshold) {
            is_vegetation = false;
        }

        if (is_vegetation && config_.enable_red_edge &&
            red_edge_start_idx >= 0 && red_edge_end_idx > red_edge_start_idx) {
            float max_deriv = compute_red_edge_derivative(
                spectrum, num_bands, red_edge_start_idx, red_edge_end_idx);
            if (max_deriv < config_.red_edge_derivative_threshold) {
                is_vegetation = false;
            }
        }

        if (is_vegetation) {
            result.full_mask[i] = -1;
            result.vegetation_indices.push_back(i);
            result.vegetation_pixels++;
        } else {
            result.filtered_pixels++;
        }
    }

    result.vegetation_ratio = num_pixels > 0
        ? static_cast<float>(result.vegetation_pixels) / num_pixels
        : 0.0f;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.filter_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

void SpectralFilter::extract_vegetation_pixels(
    const float* data,
    int num_pixels,
    int num_bands,
    const SpectralFilterResult& filter_result,
    float* output)
{
    const auto& indices = filter_result.vegetation_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
        const float* src = data + static_cast<size_t>(indices[i]) * num_bands;
        float* dst = output + i * num_bands;
        std::memcpy(dst, src, num_bands * sizeof(float));
    }
}

} // namespace hs
