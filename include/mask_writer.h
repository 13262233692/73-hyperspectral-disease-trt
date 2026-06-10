#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <gdal_priv.h>

namespace hs {

struct ClassColor {
    uint8_t r, g, b;
};

inline std::vector<ClassColor> default_class_palette(int num_classes) {
    static const ClassColor palette[] = {
        {0,   200, 0},
        {255, 200, 0},
        {200, 0,   0},
        {0,   100, 255},
        {200, 0,   200},
        {0,   200, 200},
        {128, 128, 128},
        {255, 128, 0}
    };
    std::vector<ClassColor> colors(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        if (i < 8) {
            colors[i] = palette[i];
        } else {
            colors[i] = {
                static_cast<uint8_t>((i * 53) % 256),
                static_cast<uint8_t>((i * 97) % 256),
                static_cast<uint8_t>((i * 131) % 256)
            };
        }
    }
    return colors;
}

bool save_mask_png(const std::string& path, int width, int height,
                   const int8_t* class_mask, int num_classes,
                   const std::vector<ClassColor>& palette = {});

bool save_mask_tiff(const std::string& path, int width, int height,
                    const int8_t* class_mask, int num_classes);

} // namespace hs
