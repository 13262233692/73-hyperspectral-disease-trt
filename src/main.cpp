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
        << "  --engine <path>       TensorRT engine file path\n"
        << "  --hdr <path>          ENVI .hdr file path\n"
        << "  --output <path>       Output mask PNG file path (optional)\n"
        << "  --channels <int>      Number of spectral channels (default: 200)\n"
        << "  --classes <int>       Number of classification classes (default: 3)\n"
        << "  --batch <int>         Inference batch size (default: 2048)\n"
        << "  --device <int>        CUDA device ID (default: 0)\n"
        << "  --no-normalize        Disable spectral normalization\n"
        << std::endl;
}

int main(int argc, char* argv[]) {
    std::string engine_path;
    std::string hdr_path;
    std::string output_path;
    int channels = 200;
    int classes = 3;
    int batch = 2048;
    int device = 0;
    bool normalize = true;

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
        } else if (arg == "--device" && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (arg == "--no-normalize") {
            normalize = false;
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
        std::cout << "==================================================" << std::endl;
        std::cout << "  Hyperspectral Disease Classification Engine" << std::endl;
        std::cout << "  Target: Jetson Orin (CUDA + TensorRT)" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "  Engine:   " << engine_path << std::endl;
        std::cout << "  HDR:      " << hdr_path << std::endl;
        std::cout << "  Channels: " << channels << std::endl;
        std::cout << "  Classes:  " << classes << std::endl;
        std::cout << "  Batch:    " << batch << std::endl;
        std::cout << "  Device:   " << device << std::endl;
        std::cout << "  Normalize:" << (normalize ? " yes" : " no") << std::endl;
        std::cout << "==================================================" << std::endl;

        hs::HyperspectralInferenceEngine::Config cfg;
        cfg.engine_path = engine_path;
        cfg.input_channels = channels;
        cfg.num_classes = classes;
        cfg.batch_size = batch;
        cfg.device_id = device;
        cfg.normalization.enable = normalize;

        std::cout << "\n[1/3] Initializing TensorRT engine..." << std::endl;
        hs::HyperspectralInferenceEngine engine(cfg);
        std::cout << "      Engine initialized successfully" << std::endl;

        std::cout << "\n[2/3] Loading ENVI hypercube & running inference..." << std::endl;
        auto result = engine.run(hdr_path);

        std::cout << "\n[3/3] Inference complete!" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "  Hypercube: " << result->width << " x " << result->height
                  << " x " << channels << " bands" << std::endl;
        std::cout << "  Pixels:    " << result->width * result->height << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Load:      " << result->load_time_ms << " ms" << std::endl;
        std::cout << "  Preproc:   " << result->preprocess_time_ms << " ms" << std::endl;
        std::cout << "  Inference: " << result->inference_time_ms << " ms" << std::endl;
        std::cout << "  Total:     " << result->total_time_ms << " ms" << std::endl;

        if (result->width * result->height > 0) {
            double fps = 1000.0 / result->total_time_ms;
            std::cout << "  FPS:       " << fps << std::endl;
        }
        std::cout << "--------------------------------------------------" << std::endl;

        std::vector<int> class_counts(classes, 0);
        for (auto c : result->class_mask) {
            if (c >= 0 && c < classes) class_counts[c]++;
        }
        std::cout << "\n  Class distribution:" << std::endl;
        const char* class_names[] = {"Healthy", "Yellowing", "Leaf_Blight"};
        int total_pixels = result->width * result->height;
        for (int c = 0; c < classes; ++c) {
            const char* name = (c < 3) ? class_names[c] : ("Class_" + std::to_string(c)).c_str();
            double pct = total_pixels > 0 ? (100.0 * class_counts[c] / total_pixels) : 0.0;
            std::cout << "    [" << c << "] " << std::setw(14) << std::left
                      << name << ": " << class_counts[c]
                      << " (" << std::setprecision(1) << pct << "%)" << std::endl;
        }
        std::cout << std::endl;

        if (!output_path.empty()) {
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
                std::cout << "  Color mask (PNG) saved: " << png_path << std::endl;
            } else {
                std::cerr << "  Failed to write PNG mask" << std::endl;
            }
            if (tiff_ok) {
                std::cout << "  Class mask (TIFF) saved: " << tiff_path << std::endl;
            } else {
                std::cerr << "  Failed to write TIFF mask" << std::endl;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
