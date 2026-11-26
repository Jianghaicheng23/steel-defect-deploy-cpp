#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <string>

#include "bounded_safe_queue.hpp"
#include "detector.hpp"
#include "frame_source.hpp"
#include "types.hpp"

namespace steel {

struct PipelineConfig {
    size_t queue_capacity = 8;
    OverflowPolicy overflow_policy = OverflowPolicy::kBlock;
    bool save_images = true;
    std::string output_dir = "./output";
    std::string csv_report_path;
    bool verbose = false;
};

struct PipelineStats {
    int processed_frames = 0;
    size_t dropped_stage1 = 0;
    size_t dropped_stage2 = 0;
    double wall_clock_ms = 0.0;
    double avg_read_ms = 0.0;
    double avg_preprocess_ms = 0.0;
    double avg_infer_ms = 0.0;
    double avg_postprocess_ms = 0.0;
    double avg_draw_save_ms = 0.0;
    std::array<int, kNumClasses> class_counts{};

    double fps() const {
        if (wall_clock_ms <= 0.0 || processed_frames <= 0) {
            return 0.0;
        }
        return static_cast<double>(processed_frames) * 1000.0 / wall_clock_ms;
    }
};

bool export_csv_report(const std::string& csv_path,
                       const PipelineStats& stats,
                       const std::string& mode_label,
                       bool append = false);

PipelineStats run_sync_pipeline(IDetector& detector,
                                IFrameSource& source,
                                const PipelineConfig& cfg,
                                std::atomic<bool>* external_stop = nullptr);

PipelineStats run_async_pipeline(IDetector& detector,
                                 IFrameSource& source,
                                 const PipelineConfig& cfg,
                                 std::atomic<bool>* external_stop = nullptr);

}  // namespace steel
