#include "envi_reader.h"

#include <gdal.h>
#include <cpl_conv.h>
#include <cpl_string.h>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace hs {

static Interleave parse_interleave_str(const std::string& interleave_str) {
    std::string upper = interleave_str;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "BSQ" || upper == "BAND SEQUENTIAL") return Interleave::BSQ;
    if (upper == "BIL" || upper == "BAND INTERLEAVED BY LINE") return Interleave::BIL;
    if (upper == "BIP" || upper == "BAND INTERLEAVED BY PIXEL") return Interleave::BIP;
    return Interleave::BSQ;
}

EnviStreamReader::EnviStreamReader() = default;

EnviStreamReader::~EnviStreamReader() {
    close();
}

bool EnviStreamReader::open(const std::string& hdr_path) {
    close();

    if (!std::filesystem::exists(hdr_path)) {
        return false;
    }

    dataset_.reset(GDALDataset::Open(hdr_path.c_str(), GA_ReadOnly));
    if (!dataset_) {
        return false;
    }

    width_ = dataset_->GetRasterXSize();
    height_ = dataset_->GetRasterYSize();
    bands_ = dataset_->GetRasterCount();

    if (width_ <= 0 || height_ <= 0 || bands_ <= 0) {
        dataset_.reset();
        return false;
    }

    const char* interleave_md = dataset_->GetMetadataItem("Interleave", "ENVI");
    if (interleave_md) {
        interleave_ = parse_interleave_str(interleave_md);
    } else {
        interleave_ = Interleave::BSQ;
    }

    GDALRasterBand* first_band = dataset_->GetRasterBand(1);
    data_type_ = first_band->GetRasterDataType();

    scale_.resize(bands_);
    offset_.resize(bands_);
    for (int b = 0; b < bands_; ++b) {
        GDALRasterBand* band = dataset_->GetRasterBand(b + 1);
        int has_scale, has_offset;
        double s = band->GetScale(&has_scale);
        double o = band->GetOffset(&has_offset);
        scale_[b] = has_scale ? s : 1.0;
        offset_[b] = has_offset ? o : 0.0;
    }

    line_buf_.resize(width_ * bands_);

    return true;
}

void EnviStreamReader::close() {
    dataset_.reset();
    width_ = 0;
    height_ = 0;
    bands_ = 0;
    interleave_ = Interleave::BSQ;
    data_type_ = GDT_Float32;
    scale_.clear();
    offset_.clear();
    line_buf_.clear();
}

int EnviStreamReader::read_row_block(int start_row, int num_rows, float* dst) {
    if (!dataset_ || !dst) return 0;
    if (start_row < 0 || start_row >= height_) return 0;
    if (num_rows <= 0) return 0;

    int actual_rows = std::min(num_rows, height_ - start_row);

    switch (interleave_) {
        case Interleave::BIP:
            read_row_block_bip(start_row, actual_rows, dst);
            break;
        case Interleave::BIL:
            read_row_block_bil(start_row, actual_rows, dst);
            break;
        case Interleave::BSQ:
        default:
            read_row_block_bsq(start_row, actual_rows, dst);
            break;
    }

    return actual_rows * width_;
}

void EnviStreamReader::read_row_block_bip(int start_row, int num_rows, float* dst) {
    const int w = width_;
    const int b = bands_;
    size_t pixel_base = 0;
    for (int y = 0; y < num_rows; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t dst_idx = pixel_base + static_cast<size_t>(x) * b;
            for (int band = 0; band < b; ++band) {
                GDALRasterBand* rb = dataset_->GetRasterBand(band + 1);
                float val = 0.0f;
                rb->RasterIO(GF_Read, x, start_row + y, 1, 1,
                    &val, 1, 1, GDT_Float32, 0, 0);
                dst[dst_idx + band] = static_cast<float>(val * scale_[band] + offset_[band]);
            }
        }
        pixel_base += static_cast<size_t>(w) * b;
    }
}

void EnviStreamReader::read_row_block_bil(int start_row, int num_rows, float* dst) {
    const int w = width_;
    const int b = bands_;
    std::vector<float> line(w);

    for (int y = 0; y < num_rows; ++y) {
        size_t row_base = static_cast<size_t>(y) * w * b;
        for (int band = 0; band < b; ++band) {
            GDALRasterBand* rb = dataset_->GetRasterBand(band + 1);
            rb->RasterIO(GF_Read, 0, start_row + y, w, 1,
                line.data(), w, 1, GDT_Float32, 0, 0);
            const double s = scale_[band];
            const double o = offset_[band];
            for (int x = 0; x < w; ++x) {
                dst[row_base + static_cast<size_t>(x) * b + band] =
                    static_cast<float>(line[x] * s + o);
            }
        }
    }
}

void EnviStreamReader::read_row_block_bsq(int start_row, int num_rows, float* dst) {
    const int w = width_;
    const int b = bands_;
    std::vector<float> band_block(w * num_rows);

    for (int band = 0; band < b; ++band) {
        GDALRasterBand* rb = dataset_->GetRasterBand(band + 1);
        rb->RasterIO(GF_Read, 0, start_row, w, num_rows,
            band_block.data(), w, num_rows, GDT_Float32, 0, 0);
        const double s = scale_[band];
        const double o = offset_[band];
        for (int i = 0; i < w * num_rows; ++i) {
            dst[static_cast<size_t>(i) * b + band] =
                static_cast<float>(band_block[i] * s + o);
        }
    }
}

EnviReader::EnviReader() {
    GDALAllRegister();
}

EnviReader::~EnviReader() = default;

Interleave EnviReader::parse_interleave(const std::string& interleave_str) {
    return parse_interleave_str(interleave_str);
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
    EnviStreamReader reader;
    if (!reader.open(hdr_path)) {
        throw std::runtime_error("Failed to open ENVI file: " + hdr_path);
    }

    auto cube = std::make_unique<HyperCube>();
    cube->width = reader.width();
    cube->height = reader.height();
    cube->bands = reader.bands();
    cube->interleave = reader.interleave();
    cube->data_type = reader.data_type();
    cube->data.resize(cube->total_elements());

    const int rows_per_batch = 64;
    int pixels_read = 0;
    for (int y = 0; y < cube->height; y += rows_per_batch) {
        int rows = std::min(rows_per_batch, cube->height - y);
        int n = reader.read_row_block(y, rows, cube->data.data() + pixels_read * cube->bands);
        pixels_read += n;
    }

    return cube;
}

} // namespace hs
