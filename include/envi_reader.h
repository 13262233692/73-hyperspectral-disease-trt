#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <gdal_priv.h>

namespace hs {

enum class Interleave {
    BSQ,
    BIL,
    BIP
};

struct HyperCube {
    int width;
    int height;
    int bands;
    Interleave interleave;
    GDALDataType data_type;
    std::vector<float> data;

    size_t pixel_count() const { return static_cast<size_t>(width) * height; }
    size_t total_elements() const { return static_cast<size_t>(width) * height * bands; }
    float* get_pixel_spectrum(int x, int y);
    const float* get_pixel_spectrum(int x, int y) const;
};

class EnviStreamReader {
public:
    EnviStreamReader();
    ~EnviStreamReader();

    bool open(const std::string& hdr_path);
    void close();
    bool is_open() const { return dataset_ != nullptr; }

    int width() const { return width_; }
    int height() const { return height_; }
    int bands() const { return bands_; }
    Interleave interleave() const { return interleave_; }
    GDALDataType data_type() const { return data_type_; }

    int read_row_block(int start_row, int num_rows, float* dst);

    const std::vector<double>& band_scales() const { return scale_; }
    const std::vector<double>& band_offsets() const { return offset_; }

private:
    void read_row_block_bip(int start_row, int num_rows, float* dst);
    void read_row_block_bil(int start_row, int num_rows, float* dst);
    void read_row_block_bsq(int start_row, int num_rows, float* dst);
    void load_metadata();

    GDALDatasetUniquePtr dataset_;
    int width_{0};
    int height_{0};
    int bands_{0};
    Interleave interleave_{Interleave::BSQ};
    GDALDataType data_type_{GDT_Float32};
    std::vector<double> scale_;
    std::vector<double> offset_;
    std::vector<float> line_buf_;
};

class EnviReader {
public:
    EnviReader();
    ~EnviReader();

    std::unique_ptr<HyperCube> load(const std::string& hdr_path);

private:
    static Interleave parse_interleave(const std::string& interleave_str);
    static std::string find_data_file(const std::string& hdr_path, GDALDataset* dataset);
};

} // namespace hs
