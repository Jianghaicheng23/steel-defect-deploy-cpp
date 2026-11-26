#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "async_pipeline.hpp"
#include "detector.hpp"
#include "frame_source.hpp"
#include "postprocess.hpp"
#include "types.hpp"
#include "yolov5_lite_decoder.hpp"

#ifdef USE_ORT
#include "ort_yolov5lite_detector.hpp"
#include "ort_yolov8_detector.hpp"
#endif

#ifdef USE_MNN
#include "mnn_yolov5lite_detector.hpp"
#endif

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop_requested{false};

void handle_sigint(int /*signum*/) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

enum class InputKind {
    kNone,
    kSingleImage,
    kDirectory,
    kVideoFile,
    kCamera
};

struct CliOptions {
    std::string model_path;
    std::string input_path;
    std::string video_path;
    int camera_index = -1;
    std::string output_dir = "./output";
    std::string backend = "ort_v8";
    std::string mode;  // empty = auto (sync for single image, async for batch/video/camera)
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int num_threads = 4;
    int warmup_runs = 3;
    int benchmark_runs = 1;
    size_t queue_capacity = 8;
    steel::OverflowPolicy overflow_policy = steel::OverflowPolicy::kBlock;
    bool save_images = true;
    std::string csv_path;
    int max_frames = 0;
    bool use_v5_class_thresh = false;
    bool verbose = false;
    bool show_help = false;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " --model <path> (--input <img|dir> | --video <file> | --camera <idx>) [options]\n\n"
        << "Options:\n"
        << "  --model <path>            Path to model file (.onnx / .mnn)\n"
        << "  --input <path>            Input image file or image directory\n"
        << "  --video <path>            Input video file path\n"
        << "  --camera <int>            USB camera device index\n"
        << "  --output <dir>            Directory to save annotated images (default: ./output)\n"
        << "  --backend <name>          ort|ort_v8|ort_v5lite|mnn|mnn_v5lite (default: ort_v8)\n"
        << "  --mode <sync|async|compare> Execution mode (default: auto)\n"
        << "  --conf <float>            Confidence threshold (default: 0.25)\n"
        << "  --iou <float>             NMS IoU threshold (default: 0.45)\n"
        << "  --threads <int>           Inference intra-op threads (default: 4)\n"
        << "  --warmup <int>            Warmup iterations before timing (default: 3)\n"
        << "  --runs <int>              Repeated iterations for single-image sync benchmark (default: 1)\n"
        << "  --queue-cap <int>         BoundedSafeQueue capacity in async mode (default: 8)\n"
        << "  --overflow <block|drop>   Queue backpressure policy: block or drop oldest (default: block)\n"
        << "  --save-images <0|1>       Save annotated output images (default: 1)\n"
        << "  --csv <path>              Export summary statistics to CSV file\n"
        << "  --max-frames <int>        Limit number of frames processed, 0=unlimited (default: 0)\n"
        << "  --use-v5-class-thresh     Enable NEU-DET per-class thresholds from my_detect_mnn.py\n"
        << "  --verbose                 Print per-frame detections\n"
        << "  -h, --help                Show this help message\n";
}

bool parse_args(int argc, char** argv, CliOptions& opts) {
    auto need_value = [&](int i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value after flag: " + flag);
        }
        return argv[i + 1];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
            return true;
        } else if (arg == "--model") {
            opts.model_path = need_value(i++, arg);
        } else if (arg == "--input") {
            opts.input_path = need_value(i++, arg);
        } else if (arg == "--video") {
            opts.video_path = need_value(i++, arg);
        } else if (arg == "--camera") {
            opts.camera_index = std::stoi(need_value(i++, arg));
        } else if (arg == "--output") {
            opts.output_dir = need_value(i++, arg);
        } else if (arg == "--backend") {
            opts.backend = need_value(i++, arg);
        } else if (arg == "--mode") {
            opts.mode = need_value(i++, arg);
        } else if (arg == "--conf") {
            opts.conf_threshold = std::stof(need_value(i++, arg));
        } else if (arg == "--iou") {
            opts.iou_threshold = std::stof(need_value(i++, arg));
        } else if (arg == "--threads") {
            opts.num_threads = std::stoi(need_value(i++, arg));
        } else if (arg == "--warmup") {
            opts.warmup_runs = std::stoi(need_value(i++, arg));
        } else if (arg == "--runs") {
            opts.benchmark_runs = std::stoi(need_value(i++, arg));
        } else if (arg == "--queue-cap") {
            const int cap = std::stoi(need_value(i++, arg));
            if (cap <= 0) {
                throw std::invalid_argument("--queue-cap must be > 0");
            }
            opts.queue_capacity = static_cast<size_t>(cap);
        } else if (arg == "--overflow") {
            const std::string pol = need_value(i++, arg);
            if (pol == "block") {
                opts.overflow_policy = steel::OverflowPolicy::kBlock;
            } else if (pol == "drop") {
                opts.overflow_policy = steel::OverflowPolicy::kDropOldest;
            } else {
                throw std::invalid_argument("--overflow must be 'block' or 'drop'");
            }
        } else if (arg == "--save-images") {
            const int v = std::stoi(need_value(i++, arg));
            opts.save_images = (v != 0);
        } else if (arg == "--csv") {
            opts.csv_path = need_value(i++, arg);
        } else if (arg == "--max-frames") {
            opts.max_frames = std::stoi(need_value(i++, arg));
        } else if (arg == "--use-v5-class-thresh") {
            opts.use_v5_class_thresh = true;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    const int source_count = (!opts.input_path.empty() ? 1 : 0) +
                             (!opts.video_path.empty() ? 1 : 0) +
                             (opts.camera_index >= 0 ? 1 : 0);
    if (opts.model_path.empty() || source_count == 0) {
        return false;
    }
    if (source_count > 1) {
        throw std::invalid_argument("Specify only one of --input, --video, or --camera");
    }
    if (!opts.mode.empty() &&
        opts.mode != "sync" && opts.mode != "async" && opts.mode != "compare") {
        throw std::invalid_argument("--mode must be one of: sync, async, compare");
    }
    if (opts.warmup_runs < 0 || opts.benchmark_runs <= 0 ||
        opts.num_threads <= 0 || opts.max_frames < 0) {
        throw std::invalid_argument("Invalid numeric option (--warmup >= 0, --runs/--threads > 0, --max-frames >= 0)");
    }
    return true;
}

std::string normalize_backend(const std::string& backend) {
    if (backend == "ort") {
        return "ort_v8";
    }
    if (backend == "mnn") {
        return "mnn_v5lite";
    }
    return backend;
}

std::unique_ptr<steel::IDetector> create_detector(const std::string& backend_in,
                                                  steel::DetectorConfig& cfg,
                                                  bool use_v5_class_thresh) {
    const std::string backend = normalize_backend(backend_in);
    if (backend == "ort_v8") {
#ifdef USE_ORT
        cfg.input_w = 640;
        cfg.input_h = 640;
        cfg.swap_rb = true;
        cfg.class_agnostic_nms = false;
        return std::make_unique<steel::OrtYoloV8Detector>(cfg);
#else
        throw std::runtime_error("ONNX Runtime backend was disabled at compile time (-DENABLE_ORT=OFF)");
#endif
    }
    if (backend == "ort_v5lite") {
#ifdef USE_ORT
        cfg.input_w = 320;
        cfg.input_h = 320;
        cfg.swap_rb = false;
        cfg.class_agnostic_nms = false;
        if (use_v5_class_thresh) {
            cfg.class_conf_thresholds = steel::YoloV5LiteDecoder::default_class_thresholds();
        }
        return std::make_unique<steel::OrtYoloV5LiteDetector>(cfg);
#else
        throw std::runtime_error("ONNX Runtime backend was disabled at compile time (-DENABLE_ORT=OFF)");
#endif
    }
    if (backend == "mnn_v5lite") {
#ifdef USE_MNN
        cfg.input_w = 320;
        cfg.input_h = 320;
        cfg.swap_rb = false;
        cfg.class_agnostic_nms = false;
        if (use_v5_class_thresh) {
            cfg.class_conf_thresholds = steel::YoloV5LiteDecoder::default_class_thresholds();
        }
        return std::make_unique<steel::MnnYoloV5LiteDetector>(cfg);
#else
        throw std::runtime_error(
            "MNN backend is disabled in this build. "
            "Configure CMake with -DENABLE_MNN=ON -DMNN_ROOT=<path_to_mnn> to enable it.");
#endif
    }
    throw std::invalid_argument(
        "Unsupported backend: " + backend_in +
        " (expected ort, ort_v8, ort_v5lite, mnn, or mnn_v5lite)");
}

bool is_image_file(const fs::path& p) {
    if (!fs::is_regular_file(p)) {
        return false;
    }
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> collect_images(const fs::path& input_path) {
    if (!fs::exists(input_path)) {
        throw std::runtime_error("Input path does not exist: " + input_path.string());
    }
    std::vector<fs::path> files;
    if (fs::is_regular_file(input_path)) {
        files.push_back(input_path);
        return files;
    }
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (is_image_file(entry.path())) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error("No supported image files (.jpg/.jpeg/.png/.bmp) found in: " +
                                     input_path.string());
        }
        return files;
    }
    throw std::runtime_error("Input path is neither a file nor a directory: " + input_path.string());
}

std::unique_ptr<steel::IFrameSource> make_source(InputKind kind,
                                                 const std::vector<fs::path>& images,
                                                 const CliOptions& opts) {
    if (kind == InputKind::kSingleImage || kind == InputKind::kDirectory) {
        return std::make_unique<steel::ImageListSource>(images, opts.max_frames);
    }
    if (kind == InputKind::kVideoFile) {
        return std::make_unique<steel::VideoCaptureSource>(opts.video_path, opts.max_frames);
    }
    if (kind == InputKind::kCamera) {
        return std::make_unique<steel::VideoCaptureSource>(opts.camera_index, opts.max_frames);
    }
    throw std::runtime_error("No valid input source configured");
}

void print_pipeline_stats(const std::string& title, const steel::PipelineStats& s) {
    std::cout << std::fixed << std::setprecision(2)
              << "\n=== " << title << " ===\n"
              << "  Frames      : " << s.processed_frames
              << " (dropped s1=" << s.dropped_stage1 << ", s2=" << s.dropped_stage2 << ")\n"
              << "  Wall Clock  : " << s.wall_clock_ms << " ms\n"
              << "  Throughput  : " << s.fps() << " FPS\n"
              << "  Avg Stage   : read=" << s.avg_read_ms
              << " ms | prep=" << s.avg_preprocess_ms
              << " ms | infer=" << s.avg_infer_ms
              << " ms | post=" << s.avg_postprocess_ms
              << " ms | draw+save=" << s.avg_draw_save_ms << " ms\n";

    std::cout << "  Class Counts: ";
    int total_defects = 0;
    for (int c = 0; c < steel::kNumClasses; ++c) {
        total_defects += s.class_counts[c];
        std::cout << steel::kClassNames[c] << "=" << s.class_counts[c]
                  << (c + 1 < steel::kNumClasses ? " | " : "");
    }
    std::cout << " (Total=" << total_defects << ")\n";
}

void print_compare_table(const steel::PipelineStats& sync_s,
                         const steel::PipelineStats& async_s) {
    const double speedup = (async_s.wall_clock_ms > 0.0)
                               ? (sync_s.wall_clock_ms / async_s.wall_clock_ms)
                               : 0.0;
    const bool counts_match = (sync_s.processed_frames == async_s.processed_frames) &&
                              (sync_s.class_counts == async_s.class_counts);

    std::cout << std::fixed << std::setprecision(2)
              << "\n=== Sync vs 3-Thread Async Comparison ===\n"
              << "  Sync  : " << sync_s.processed_frames << " frames, "
              << sync_s.wall_clock_ms << " ms, " << sync_s.fps() << " FPS\n"
              << "  Async : " << async_s.processed_frames << " frames, "
              << async_s.wall_clock_ms << " ms, " << async_s.fps() << " FPS\n"
              << "  Speedup     : " << speedup << "x\n"
              << "  Consistency : " << (counts_match ? "PASS (exact match)" : "MISMATCH") << "\n";
}

void run_single_image_benchmark(steel::IDetector& detector,
                                const fs::path& img_path,
                                const CliOptions& opts) {
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to read input image: " + img_path.string());
    }

    double sum_prep = 0.0;
    double sum_infer = 0.0;
    double sum_post = 0.0;
    double sum_total = 0.0;
    double min_total = std::numeric_limits<double>::max();
    double max_total = 0.0;

    steel::InferResult last_res;
    for (int r = 0; r < opts.benchmark_runs; ++r) {
        last_res = detector.infer(img);
        const double tot = last_res.timing.total_ms();
        sum_prep += last_res.timing.preprocess_ms;
        sum_infer += last_res.timing.infer_ms;
        sum_post += last_res.timing.postprocess_ms;
        sum_total += tot;
        min_total = std::min(min_total, tot);
        max_total = std::max(max_total, tot);
    }

    std::cout << "[" << img_path.filename().string() << "] Detections: "
              << last_res.detections.size() << "\n";
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < last_res.detections.size(); ++i) {
        const auto& d = last_res.detections[i];
        const std::string cls_name =
            (d.class_id >= 0 && d.class_id < steel::kNumClasses)
                ? steel::kClassNames[d.class_id]
                : "unknown";
        std::cout << "  #" << i + 1 << " [" << cls_name << "] score=" << d.score
                  << " box=(" << std::setprecision(1)
                  << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << ")"
                  << std::setprecision(3) << "\n";
    }

    if (opts.save_images) {
        fs::create_directories(opts.output_dir);
        std::array<int, steel::kNumClasses> img_counts{};
        cv::Mat vis = steel::draw_detections(img, last_res.detections, img_counts);
        const fs::path out_path = fs::path(opts.output_dir) / img_path.filename();
        if (!cv::imwrite(out_path.string(), vis)) {
            throw std::runtime_error("Failed to write output image: " + out_path.string());
        }
        std::cout << "  Saved annotated image: " << out_path.string() << "\n";
    }

    const double n = static_cast<double>(opts.benchmark_runs);
    const double avg_total = sum_total / n;
    const double fps = avg_total > 0.0 ? (1000.0 / avg_total) : 0.0;
    std::cout << std::fixed << std::setprecision(2)
              << "\n=== Latency Breakdown [" << detector.backend_name()
              << "] (" << opts.benchmark_runs << " runs) ===\n"
              << "  Preprocess  : " << (sum_prep / n) << " ms\n"
              << "  Inference   : " << (sum_infer / n) << " ms\n"
              << "  Postprocess : " << (sum_post / n) << " ms\n"
              << "  Total Avg   : " << avg_total << " ms"
              << "  (min=" << min_total << " ms, max=" << max_total << " ms)\n"
              << "  Throughput  : " << fps << " FPS\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    try {
        CliOptions opts;
        if (!parse_args(argc, argv, opts)) {
            print_usage(argv[0]);
            return opts.show_help ? 0 : 1;
        }
        if (opts.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        steel::DetectorConfig cfg;
        cfg.model_path = opts.model_path;
        cfg.conf_threshold = opts.conf_threshold;
        cfg.iou_threshold = opts.iou_threshold;
        cfg.num_threads = opts.num_threads;

        std::unique_ptr<steel::IDetector> detector =
            create_detector(opts.backend, cfg, opts.use_v5_class_thresh);
        detector->print_model_info();

        InputKind kind = InputKind::kNone;
        std::vector<fs::path> images;
        if (!opts.input_path.empty()) {
            images = collect_images(opts.input_path);
            kind = fs::is_regular_file(opts.input_path)
                       ? InputKind::kSingleImage
                       : InputKind::kDirectory;
        } else if (!opts.video_path.empty()) {
            if (!fs::exists(opts.video_path)) {
                throw std::runtime_error("Video file does not exist: " + opts.video_path);
            }
            kind = InputKind::kVideoFile;
        } else if (opts.camera_index >= 0) {
            kind = InputKind::kCamera;
        }

        std::string mode = opts.mode;
        if (mode.empty()) {
            mode = (kind == InputKind::kSingleImage) ? "sync" : "async";
        }

        if (opts.warmup_runs > 0) {
            cv::Mat warmup_img;
            if (!images.empty()) {
                warmup_img = cv::imread(images.front().string(), cv::IMREAD_COLOR);
                if (warmup_img.empty()) {
                    throw std::runtime_error("Failed to read warmup image: " +
                                             images.front().string());
                }
            } else {
                warmup_img = cv::Mat(200, 200, CV_8UC3, cv::Scalar(114, 114, 114));
            }
            for (int w = 0; w < opts.warmup_runs; ++w) {
                (void)detector->infer(warmup_img);
            }
        }

        if (kind == InputKind::kSingleImage && mode == "sync" && opts.csv_path.empty()) {
            run_single_image_benchmark(*detector, images.front(), opts);
            return 0;
        }

        steel::PipelineConfig pipe_cfg;
        pipe_cfg.queue_capacity = opts.queue_capacity;
        pipe_cfg.overflow_policy = opts.overflow_policy;
        pipe_cfg.save_images = opts.save_images;
        pipe_cfg.output_dir = opts.output_dir;
        pipe_cfg.csv_report_path = opts.csv_path;
        pipe_cfg.verbose = opts.verbose;

        if (mode == "sync") {
            auto src = make_source(kind, images, opts);
            steel::PipelineStats stats =
                steel::run_sync_pipeline(*detector, *src, pipe_cfg, &g_stop_requested);
            print_pipeline_stats("Sync Pipeline [" + detector->backend_name() + "]", stats);
            if (!opts.csv_path.empty()) {
                steel::export_csv_report(opts.csv_path, stats, "sync", false);
                std::cout << "  CSV report saved: " << opts.csv_path << "\n";
            }
        } else if (mode == "async") {
            auto src = make_source(kind, images, opts);
            steel::PipelineStats stats =
                steel::run_async_pipeline(*detector, *src, pipe_cfg, &g_stop_requested);
            print_pipeline_stats("3-Thread Async Pipeline [" + detector->backend_name() + "]", stats);
            if (!opts.csv_path.empty()) {
                steel::export_csv_report(opts.csv_path, stats, "async", false);
                std::cout << "  CSV report saved: " << opts.csv_path << "\n";
            }
        } else {
            auto src_sync = make_source(kind, images, opts);
            steel::PipelineStats sync_stats =
                steel::run_sync_pipeline(*detector, *src_sync, pipe_cfg, &g_stop_requested);
            print_pipeline_stats("Sync Baseline [" + detector->backend_name() + "]", sync_stats);

            if (!g_stop_requested.load(std::memory_order_relaxed)) {
                auto src_async = make_source(kind, images, opts);
                steel::PipelineStats async_stats =
                    steel::run_async_pipeline(*detector, *src_async, pipe_cfg, &g_stop_requested);
                print_pipeline_stats("3-Thread Async [" + detector->backend_name() + "]", async_stats);
                print_compare_table(sync_stats, async_stats);

                if (!opts.csv_path.empty()) {
                    steel::export_csv_report(opts.csv_path, sync_stats, "sync", false);
                    steel::export_csv_report(opts.csv_path, async_stats, "async", true);
                    std::cout << "  CSV comparison saved: " << opts.csv_path << "\n";
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
