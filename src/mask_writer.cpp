#include "mask_writer.h"

#include <gdal_priv.h>
#include <cpl_conv.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <filesystem>

namespace hs {

static void ensure_gdal_registered() {
    static bool registered = false;
    if (!registered) {
        GDALAllRegister();
        registered = true;
    }
}

bool save_mask_png(const std::string& path, int width, int height,
                   const int8_t* class_mask, int num_classes,
                   const std::vector<ClassColor>& palette) {
    if (!class_mask || width <= 0 || height <= 0) return false;
    ensure_gdal_registered();

    std::vector<ClassColor> colors = palette;
    if (colors.empty()) {
        colors = default_class_palette(num_classes);
    }

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("PNG");
    if (!driver) return false;

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");

    GDALDataset* ds = driver->Create(path.c_str(), width, height, 3, GDT_Byte, options);
    CSLDestroy(options);
    if (!ds) return false;

    std::vector<uint8_t> r_buf(width * height);
    std::vector<uint8_t> g_buf(width * height);
    std::vector<uint8_t> b_buf(width * height);

    for (int i = 0; i < width * height; ++i) {
        int cls = class_mask[i];
        if (cls < 0 || cls >= num_classes) cls = 0;
        r_buf[i] = colors[cls].r;
        g_buf[i] = colors[cls].g;
        b_buf[i] = colors[cls].b;
    }

    ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        r_buf.data(), width, height, GDT_Byte, 0, 0);
    ds->GetRasterBand(2)->RasterIO(GF_Write, 0, 0, width, height,
        g_buf.data(), width, height, GDT_Byte, 0, 0);
    ds->GetRasterBand(3)->RasterIO(GF_Write, 0, 0, width, height,
        b_buf.data(), width, height, GDT_Byte, 0, 0);

    GDALClose(ds);
    return true;
}

bool save_mask_tiff(const std::string& path, int width, int height,
                    const int8_t* class_mask, int num_classes) {
    if (!class_mask || width <= 0 || height <= 0) return false;
    ensure_gdal_registered();

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) return false;

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");

    GDALDataset* ds = driver->Create(path.c_str(), width, height, 1, GDT_Byte, options);
    CSLDestroy(options);
    if (!ds) return false;

    std::vector<uint8_t> buf(width * height);
    for (int i = 0; i < width * height; ++i) {
        int cls = class_mask[i];
        if (cls < 0 || cls >= num_classes) cls = 0;
        buf[i] = static_cast<uint8_t>(cls);
    }

    ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        buf.data(), width, height, GDT_Byte, 0, 0);
    ds->GetRasterBand(1)->SetNoDataValue(255.0);

    GDALClose(ds);
    return true;
}

} // namespace hs
