#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include "hyperspectral_engine.h"
#include "mask_writer.h"

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
        << "  --engine <path>            TensorRT engine file path\n"
        << "  --hdr <path>               ENVI .hdr file path\n"
        << "  --output <path>            Output mask PNG file path (optional)\n"
        << "  --channels <int>           Number of spectral channels (default: 200)\n"
        << "  --classes <int>            Number of classification classes (default: 3)\n"
        << "  --batch <int>              Inference batch size per stream (default: 4096)\n"
        << "  --streams <int>            Number of CUDA streams for overlap (default: 2)\n"
        << "  --rows-per-tile <int>      Rows per streaming tile (default: 128)\n"
        << "  --device <int>             CUDA device ID (default: 0)\n"
        << "  --no-normalize             Disable spectral normalization\n"
        << "  --no-streaming             Disable streaming mode (load full cube)\n"
        << "  --no-filter                Disable NDVI + Red Edge vegetation filter\n"
        << "  --ndvi-threshold <float>   NDVI threshold for vegetation (default: 0.3)\n"
        << "  --red-edge-thresh <float>  Red Edge derivative threshold (default: 0.005)\n"
        << "  --nir-wavelength <float>   NIR wavelength in nm (default: 800)\n"
        << "  --red-wavelength <float>   Red wavelength in nm (default: 680)\n"
        << "  --non-veg-label <int>      Label for non-vegetation pixels (default: 0)\n"
        << std::endl;
}

int main(int argc, char* argv[]) {
    std::string engine_path;
    std::string hdr_path;
    std::string output_path;
    int channels = 200;
    int classes = 3;
    int batch = 4096;
    int streams = 2;
    int rows_per_tile = 128;
    int device = 0;
    bool normalize = true;
    bool use_streaming = true;
    bool enable_filter = true;
    float ndvi_threshold = 0.3f;
    float red_edge_thresh = 0.005f;
    float nir_wavelength = 800.0f;
    float red_wavelength = 680.0f;
    int non_veg_label = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--hdr" && i + 1 < argc) {
            hdr_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--channels" && i + 1 < argc) {
            channels = std::atoi(argv[++i]);
        } else if (arg == "--classes" && i + 1 < argc) {
            classes = std::atoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        } else if (arg == "--streams" && i + 1 < argc) {
            streams = std::atoi(argv[++i]);
        } else if (arg == "--rows-per-tile" && i + 1 < argc) {
            rows_per_tile = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (arg == "--no-normalize") {
            normalize = false;
        } else if (arg == "--no-streaming") {
            use_streaming = false;
        } else if (arg == "--no-filter") {
            enable_filter = false;
        } else if (arg == "--ndvi-threshold" && i + 1 < argc) {
            ndvi_threshold = std::stof(argv[++i]);
        } else if (arg == "--red-edge-thresh" && i + 1 < argc) {
            red_edge_thresh = std::stof(argv[++i]);
        } else if (arg == "--nir-wavelength" && i + 1 < argc) {
            nir_wavelength = std::stof(argv[++i]);
        } else if (arg == "--red-wavelength" && i + 1 < argc) {
            red_wavelength = std::stof(argv[++i]);
        } else if (arg == "--non-veg-label" && i + 1 < argc) {
            non_veg_label = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (engine_path.empty() || hdr_path.empty()) {
        std::cerr << "Error: --engine and --hdr are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "========================================================" << std::endl;
        std::cout << "  Hyperspectral Disease Classification Engine" << std::endl;
        std::cout << "  Target: Jetson Orin (CUDA + TRT + Pinned + NDVI)" << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout << "  Engine:           " << engine_path << std::endl;
        std::cout << "  HDR:              " << hdr_path << std::endl;
        std::cout << "  Channels:         " << channels << std::endl;
        std::cout << "  Classes:          " << classes << std::endl;
        std::cout << "  Batch/stream:     " << batch << " pixels" << std::endl;
        std::cout << "  CUDA streams:     " << streams << " (pinned + overlap)" << std::endl;
        std::cout << "  Rows/tile:        " << rows_per_tile << std::endl;
        std::cout << "  Device:           " << device << std::endl;
        std::cout << "  Normalize:        " << (normalize ? "yes" : "no") << std::endl;
        std::cout << "  Streaming:        " << (use_streaming ? "yes" : "no") << std::endl;
        std::cout << "  Vegetation Filter:" << (enable_filter ? "yes" : "no") << std::endl;
        if (enable_filter) {
            std::cout << "    NDVI threshold: " << ndvi_threshold << std::endl;
            std::cout << "    RedEdge thresh: " << red_edge_thresh << std::endl;
            std::cout << "    NIR wavelength: " << nir_wavelength << " nm" << std::endl;
            std::cout << "    Red wavelength: " << red_wavelength << " nm" << std::endl;
            std::cout << "    Non-veg label:  " << non_veg_label << std::endl;
        }
        std::cout << "========================================================" << std::endl;

        hs::HyperspectralInferenceEngine::Config cfg;
        cfg.engine_path = engine_path;
        cfg.input_channels = channels;
        cfg.num_classes = classes;
        cfg.batch_size = batch;
        cfg.num_streams = streams;
        cfg.rows_per_tile = rows_per_tile;
        cfg.device_id = device;
        cfg.normalization.enable = normalize;
        cfg.use_streaming = use_streaming;
        cfg.enable_spectral_filter = enable_filter;
        cfg.filter_config.ndvi_threshold = ndvi_threshold;
        cfg.filter_config.red_edge_derivative_threshold = red_edge_thresh;
        cfg.filter_config.nir_wavelength_nm = nir_wavelength;
        cfg.filter_config.red_wavelength_nm = red_wavelength;
        cfg.filter_config.non_vegetation_label = static_cast<int8_t>(non_veg_label);

        std::cout << "\n[1/4] Initializing TensorRT engine + Pinned Memory..." << std::endl;
        hs::HyperspectralInferenceEngine engine(cfg);

        {
            size_t input_mb_per_stream = static_cast<size_t>(batch) * channels * sizeof(float) / (1024 * 1024);
            std::cout << "      Pinned host memory: ~" << input_mb_per_stream * streams
                      << " MB (" << streams << " streams)" << std::endl;
            if (enable_filter) {
                std::cout << "      Spectral filter:    NDVI + Red Edge (physics pre-filter)" << std::endl;
            }
            std::cout << "      Engine initialized successfully" << std::endl;
        }

        std::cout << "\n[2/4] Streaming ENVI hypercube through pipeline..." << std::endl;
        auto result = engine.run(hdr_path);

        std::cout << "\n[3/4] Inference pipeline complete!" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "  Hypercube:     " << result->width << " x " << result->height
                  << " x " << channels << " bands" << std::endl;
        std::cout << "  Total pixels:  " << result->total_pixels << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        if (enable_filter) {
            std::cout << "  Vegetation:    " << result->vegetation_pixels
                      << " (" << std::setprecision(1)
                      << (result->vegetation_ratio * 100.0f) << "%)" << std::endl;
            std::cout << "  Filtered out:  " << result->filtered_pixels
                      << " (" << std::setprecision(1)
                      << ((1.0f - result->vegetation_ratio) * 100.0f) << "%)" << std::endl;
            std::cout << "  Filter time:   " << std::setprecision(2)
                      << result->filter_time_ms << " ms" << std::endl;
            double saved_pct = result->total_pixels > 0
                ? 100.0 * result->filtered_pixels / result->total_pixels : 0.0;
            std::cout << "  GPU saved:     ~" << std::setprecision(0) << saved_pct
                      << "% inference workload" << std::endl;
        }
        std::cout << std::setprecision(2);
        std::cout << "  Load:          " << result->load_time_ms << " ms" << std::endl;
        std::cout << "  Preprocess:    " << result->preprocess_time_ms << " ms" << std::endl;
        std::cout << "  Inference:     " << result->inference_time_ms << " ms" << std::endl;
        std::cout << "  Total:         " << result->total_time_ms << " ms" << std::endl;

        if (result->total_pixels > 0) {
            double fps = 1000.0 / result->total_time_ms;
            double mpps = static_cast<double>(result->total_pixels) / result->total_time_ms / 1000.0;
            std::cout << "  FPS:           " << fps << std::endl;
            std::cout << "  Throughput:    " << mpps << " Mpix/s" << std::endl;
        }
        std::cout << "--------------------------------------------------------" << std::endl;

        std::vector<int> class_counts(classes, 0);
        int non_veg_count = 0;
        for (auto c : result->class_mask) {
            if (c == non_veg_label && enable_filter) {
                non_veg_count++;
            } else if (c >= 0 && c < classes) {
                class_counts[c]++;
            }
        }

        std::cout << "\n  Classification breakdown:" << std::endl;
        const char* class_names[] = {"Healthy", "Yellowing", "Leaf_Blight"};
        for (int c = 0; c < classes; ++c) {
            const char* name = (c < 3) ? class_names[c] : ("Class_" + std::to_string(c)).c_str();
            double pct = result->total_pixels > 0
                ? 100.0 * class_counts[c] / result->total_pixels : 0.0;
            std::cout << "    [" << c << "] " << std::setw(14) << std::left
                      << name << ": " << std::setw(8) << std::right << class_counts[c]
                      << " (" << std::setprecision(1) << pct << "%)" << std::endl;
        }
        if (enable_filter && non_veg_count > 0) {
            double pct = 100.0 * non_veg_count / result->total_pixels;
            std::cout << "    [X] " << std::setw(14) << std::left
                      << "Non-Vegetation" << ": " << std::setw(8) << std::right << non_veg_count
                      << " (" << std::setprecision(1) << pct << "%)" << std::endl;
        }
        std::cout << std::endl;

        if (!output_path.empty()) {
            std::cout << "[4/4] Writing output masks..." << std::endl;

            std::string png_path = output_path;
            std::string tiff_path;
            auto pos = output_path.find_last_of('.');
            if (pos != std::string::npos) {
                tiff_path = output_path.substr(0, pos) + ".tif";
            } else {
                tiff_path = output_path + ".tif";
                png_path = output_path + ".png";
            }

            bool png_ok = hs::save_mask_png(png_path, result->width, result->height,
                result->class_mask.data(), classes);
            bool tiff_ok = hs::save_mask_tiff(tiff_path, result->width, result->height,
                result->class_mask.data(), classes);

            if (png_ok) {
                std::cout << "    Color mask (PNG):  " << png_path << std::endl;
            } else {
                std::cerr << "    Failed to write PNG mask" << std::endl;
            }
            if (tiff_ok) {
                std::cout << "    Class mask (TIFF): " << tiff_path << std::endl;
            } else {
                std::cerr << "    Failed to write TIFF mask" << std::endl;
            }
        }

        std::cout << "\nAll done. Exiting." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
