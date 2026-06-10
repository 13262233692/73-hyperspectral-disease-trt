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

class EnviReader {
public:
    EnviReader();
    ~EnviReader();

    std::unique_ptr<HyperCube> load(const std::string& hdr_path);

private:
    static Interleave parse_interleave(const std::string& interleave_str);
    static std::string find_data_file(const std::string& hdr_path, GDALDataset* dataset);
    void convert_to_contiguous_float(GDALRasterBand* band, int band_idx, HyperCube& cube);
};

} // namespace hs
