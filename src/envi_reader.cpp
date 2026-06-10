#include "envi_reader.h"

#include <gdal.h>
#include <cpl_conv.h>
#include <cpl_string.h>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace hs {

EnviReader::EnviReader() {
    GDALAllRegister();
}

EnviReader::~EnviReader() = default;

Interleave EnviReader::parse_interleave(const std::string& interleave_str) {
    std::string upper = interleave_str;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "BSQ" || upper == "BAND SEQUENTIAL") return Interleave::BSQ;
    if (upper == "BIL" || upper == "BAND INTERLEAVED BY LINE") return Interleave::BIL;
    if (upper == "BIP" || upper == "BAND INTERLEAVED BY PIXEL") return Interleave::BIP;
    throw std::runtime_error("Unknown interleave format: " + interleave_str);
}

std::string EnviReader::find_data_file(const std::string& hdr_path, GDALDataset* dataset) {
    const char* data_file = dataset->GetMetadataItem("Data File", "ENVI");
    if (data_file && std::filesystem::exists(data_file)) {
        return data_file;
    }
    std::filesystem::path hdr(hdr_path);
    auto base = hdr.parent_path() / hdr.stem();
    std::vector<std::string> candidates = {
        base.string() + ".img",
        base.string() + ".dat",
        base.string() + ".raw",
        base.string() + ".cube",
        base.string()
    };
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) return c;
    }
    throw std::runtime_error("Cannot find ENVI data file for: " + hdr_path);
}

float* HyperCube::get_pixel_spectrum(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return nullptr;
    size_t idx = (static_cast<size_t>(y) * width + x) * bands;
    return data.data() + idx;
}

const float* HyperCube::get_pixel_spectrum(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) return nullptr;
    size_t idx = (static_cast<size_t>(y) * width + x) * bands;
    return data.data() + idx;
}

std::unique_ptr<HyperCube> EnviReader::load(const std::string& hdr_path) {
    if (!std::filesystem::exists(hdr_path)) {
        throw std::runtime_error("HDR file not found: " + hdr_path);
    }

    GDALDatasetUniquePtr dataset(GDALDataset::Open(hdr_path.c_str(), GA_ReadOnly));
    if (!dataset) {
        throw std::runtime_error("Failed to open ENVI file: " + hdr_path);
    }

    auto cube = std::make_unique<HyperCube>();
    cube->width = dataset->GetRasterXSize();
    cube->height = dataset->GetRasterYSize();
    cube->bands = dataset->GetRasterCount();

    if (cube->width <= 0 || cube->height <= 0 || cube->bands <= 0) {
        throw std::runtime_error("Invalid hypercube dimensions");
    }

    const char* interleave_md = dataset->GetMetadataItem("Interleave", "ENVI");
    if (interleave_md) {
        cube->interleave = parse_interleave(interleave_md);
    } else {
        cube->interleave = Interleave::BSQ;
    }

    GDALRasterBand* first_band = dataset->GetRasterBand(1);
    cube->data_type = first_band->GetRasterDataType();

    cube->data.resize(cube->total_elements());

    std::vector<double> scale(cube->bands, 1.0);
    std::vector<double> offset(cube->bands, 0.0);
    for (int b = 0; b < cube->bands; ++b) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
        int has_scale, has_offset;
        scale[b] = band->GetScale(&has_scale);
        offset[b] = band->GetOffset(&has_offset);
        if (!has_scale) scale[b] = 1.0;
        if (!has_offset) offset[b] = 0.0;
    }

    if (cube->interleave == Interleave::BIP) {
        for (int y = 0; y < cube->height; ++y) {
            for (int x = 0; x < cube->width; ++x) {
                size_t pixel_base = (static_cast<size_t>(y) * cube->width + x) * cube->bands;
                for (int b = 0; b < cube->bands; ++b) {
                    GDALRasterBand* band = dataset->GetRasterBand(b + 1);
                    float val = 0.0f;
                    band->RasterIO(GF_Read, x, y, 1, 1, &val, 1, 1, GDT_Float32, 0, 0);
                    cube->data[pixel_base + b] = static_cast<float>(val * scale[b] + offset[b]);
                }
            }
        }
    } else if (cube->interleave == Interleave::BIL) {
        for (int y = 0; y < cube->height; ++y) {
            std::vector<float> line_buffer(cube->width);
            for (int b = 0; b < cube->bands; ++b) {
                GDALRasterBand* band = dataset->GetRasterBand(b + 1);
                band->RasterIO(GF_Read, 0, y, cube->width, 1,
                    line_buffer.data(), cube->width, 1, GDT_Float32, 0, 0);
                for (int x = 0; x < cube->width; ++x) {
                    size_t idx = (static_cast<size_t>(y) * cube->width + x) * cube->bands + b;
                    cube->data[idx] = static_cast<float>(line_buffer[x] * scale[b] + offset[b]);
                }
            }
        }
    } else {
        std::vector<float> band_buffer(cube->width * cube->height);
        for (int b = 0; b < cube->bands; ++b) {
            GDALRasterBand* band = dataset->GetRasterBand(b + 1);
            band->RasterIO(GF_Read, 0, 0, cube->width, cube->height,
                band_buffer.data(), cube->width, cube->height, GDT_Float32, 0, 0);
            for (size_t i = 0; i < band_buffer.size(); ++i) {
                size_t dst_idx = i * cube->bands + b;
                cube->data[dst_idx] = static_cast<float>(band_buffer[i] * scale[b] + offset[b]);
            }
        }
    }

    return cube;
}

} // namespace hs
