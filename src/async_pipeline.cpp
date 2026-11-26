#include "async_pipeline.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <thread>
#include <utility>

#include "postprocess.hpp"

namespace fs = std::filesystem;

namespace steel {

namespace {

struct PrepTask {
    int frame_id = 0;
    std::string name;
    cv::Mat orig_image;
    PreprocessedData prep;
    double read_ms = 0.0;
    double prep_ms = 0.0;
};

struct InferTask {
    int frame_id = 0;
    std::string name;
    cv::Mat orig_image;
    LetterBoxInfo lb_info;
    RawOutput raw;
    double read_ms = 0.0;
    double prep_ms = 0.0;
    double infer_ms = 0.0;
};

class ThreadJoinGuard {
public:
    explicit ThreadJoinGuard(std::thread& t) : thread_(t) {}
    ~ThreadJoinGuard() {
        if (thread_.joinable()) {
            thread_.join();//析构自动收尾
        }
    }
    ThreadJoinGuard(const ThreadJoinGuard&) = delete;
    ThreadJoinGuard& operator=(const ThreadJoinGuard&) = delete;

private:
    std::thread& thread_;
};

bool stop_requested(const std::atomic<bool>& abort_flag,
                    const std::atomic<bool>* external_stop) {
    if (abort_flag.load(std::memory_order_relaxed)) {
        return true;
    }
    return external_stop != nullptr && external_stop->load(std::memory_order_relaxed);
}

void log_frame_detections(const std::string& name, const std::vector<Detection>& dets) {
    std::cout << "[" << name << "] Detections: " << dets.size() << "\n";
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        const std::string cls_name =
            (d.class_id >= 0 && d.class_id < kNumClasses) ? kClassNames[d.class_id] : "unknown";
        std::cout << "  #" << i + 1 << " [" << cls_name << "] score=" << d.score
                  << " box=(" << std::setprecision(1)
                  << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << ")"
                  << std::setprecision(3) << "\n";
    }
}

void finalize_averages(PipelineStats& stats,
                       double sum_read,
                       double sum_prep,
                       double sum_infer,
                       double sum_post,
                       double sum_draw_save) {
    if (stats.processed_frames <= 0) {
        return;
    }
    const double n = static_cast<double>(stats.processed_frames);
    stats.avg_read_ms = sum_read / n;
    stats.avg_preprocess_ms = sum_prep / n;
    stats.avg_infer_ms = sum_infer / n;
    stats.avg_postprocess_ms = sum_post / n;
    stats.avg_draw_save_ms = sum_draw_save / n;
}

}  // namespace

bool export_csv_report(const std::string& csv_path,
                       const PipelineStats& stats,
                       const std::string& mode_label,
                       bool append) {
    if (csv_path.empty()) {
        return false;
    }

    const fs::path p(csv_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    const bool write_header = !append || !fs::exists(p) || fs::file_size(p) == 0;
    std::ofstream ofs(csv_path, append ? (std::ios::out | std::ios::app)
                                       : (std::ios::out | std::ios::trunc));
    if (!ofs.is_open()) {
        return false;
    }

    if (write_header) {
        ofs << "mode,processed_frames,wall_clock_ms,fps,"
            << "avg_read_ms,avg_preprocess_ms,avg_infer_ms,avg_postprocess_ms,avg_draw_save_ms,"
            << "dropped_stage1,dropped_stage2";
        for (const auto& cls : kClassNames) {
            ofs << "," << cls;
        }
        ofs << ",total_defects\n";
    }

    int total_defects = 0;
    for (int c : stats.class_counts) {
        total_defects += c;
    }

    ofs << std::fixed << std::setprecision(2)
        << mode_label << ","
        << stats.processed_frames << ","
        << stats.wall_clock_ms << ","
        << stats.fps() << ","
        << stats.avg_read_ms << ","
        << stats.avg_preprocess_ms << ","
        << stats.avg_infer_ms << ","
        << stats.avg_postprocess_ms << ","
        << stats.avg_draw_save_ms << ","
        << stats.dropped_stage1 << ","
        << stats.dropped_stage2;
    for (int c : stats.class_counts) {
        ofs << "," << c;
    }
    ofs << "," << total_defects << "\n";
    return ofs.good();
}

PipelineStats run_sync_pipeline(IDetector& detector,
                                IFrameSource& source,
                                const PipelineConfig& cfg,
                                std::atomic<bool>* external_stop) {
    using Clock = std::chrono::steady_clock;

    if (cfg.save_images && !cfg.output_dir.empty()) {
        fs::create_directories(cfg.output_dir);
    }

    PipelineStats stats;
    stats.class_counts.fill(0);

    double sum_read = 0.0;
    double sum_prep = 0.0;
    double sum_infer = 0.0;
    double sum_post = 0.0;
    double sum_draw_save = 0.0;

    const auto wall_start = Clock::now();

    FramePacket pkt;
    while (external_stop == nullptr || !external_stop->load(std::memory_order_relaxed)) {
        if (!source.next(pkt)) {
            break;
        }

        const auto t0 = Clock::now();
        PreprocessedData prep = detector.preprocess(pkt.image);
        const auto t1 = Clock::now();
        RawOutput raw = detector.run_raw(prep.blob);
        const auto t2 = Clock::now();
        std::vector<Detection> dets = detector.decode_and_postprocess(raw, prep.lb_info);
        const auto t3 = Clock::now();

        std::array<int, kNumClasses> frame_counts{};
        frame_counts.fill(0);
        if (cfg.save_images) {
            cv::Mat vis = draw_detections(pkt.image, dets, frame_counts);
            const fs::path out_path = fs::path(cfg.output_dir) / pkt.name;
            if (!cv::imwrite(out_path.string(), vis)) {
                throw std::runtime_error("Failed to write output image: " + out_path.string());
            }
        } else {
            for (const auto& d : dets) {
                if (d.class_id >= 0 && d.class_id < kNumClasses) {
                    frame_counts[d.class_id] += 1;
                }
            }
        }
        const auto t4 = Clock::now();

        if (cfg.verbose) {
            log_frame_detections(pkt.name, dets);
        }

        for (int c = 0; c < kNumClasses; ++c) {
            stats.class_counts[c] += frame_counts[c];
        }
        stats.processed_frames += 1;

        sum_read += pkt.read_ms;
        sum_prep += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_infer += std::chrono::duration<double, std::milli>(t2 - t1).count();
        sum_post += std::chrono::duration<double, std::milli>(t3 - t2).count();
        sum_draw_save += std::chrono::duration<double, std::milli>(t4 - t3).count();
    }

    const auto wall_end = Clock::now();
    stats.wall_clock_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    finalize_averages(stats, sum_read, sum_prep, sum_infer, sum_post, sum_draw_save);
    return stats;
}

PipelineStats run_async_pipeline(IDetector& detector,
                                 IFrameSource& source,
                                 const PipelineConfig& cfg,
                                 std::atomic<bool>* external_stop) {
    using Clock = std::chrono::steady_clock;

    if (cfg.save_images && !cfg.output_dir.empty()) {
        fs::create_directories(cfg.output_dir);
    }

    BoundedSafeQueue<PrepTask> q_prep_to_infer(cfg.queue_capacity, cfg.overflow_policy);
    BoundedSafeQueue<InferTask> q_infer_to_post(cfg.queue_capacity, cfg.overflow_policy);

    std::atomic<bool> abort_flag{false};
    std::exception_ptr exc_t1 = nullptr;
    std::exception_ptr exc_t2 = nullptr;
    std::exception_ptr exc_t3 = nullptr;

    PipelineStats stats;
    stats.class_counts.fill(0);

    double sum_read = 0.0;
    double sum_prep = 0.0;
    double sum_infer = 0.0;
    double sum_post = 0.0;
    double sum_draw_save = 0.0;

    const auto wall_start = Clock::now();

    std::thread t1_prep;
    std::thread t2_infer;
    std::thread t3_post;

    {
        ThreadJoinGuard guard_t1(t1_prep);
        ThreadJoinGuard guard_t2(t2_infer);
        ThreadJoinGuard guard_t3(t3_post);

        // Stage 1: Frame capture + letterbox + HWC->CHW normalize
        t1_prep = std::thread([&]() {
            try {
                FramePacket pkt;
                while (!stop_requested(abort_flag, external_stop)) {
                    if (!source.next(pkt)) {
                        break;
                    }
                    const auto t0 = Clock::now();
                    PreprocessedData prep = detector.preprocess(pkt.image);
                    const auto t1 = Clock::now();

                    PrepTask task;
                    task.frame_id = pkt.frame_id;
                    task.name = std::move(pkt.name);
                    task.orig_image = std::move(pkt.image);
                    task.prep = std::move(prep);
                    task.read_ms = pkt.read_ms;
                    task.prep_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    if (!q_prep_to_infer.push(std::move(task))) {
                        break;
                    }
                }
            } catch (...) {
                exc_t1 = std::current_exception();
                abort_flag.store(true, std::memory_order_relaxed);
                q_infer_to_post.close();
            }
            q_prep_to_infer.close();
        });

        // Stage 2: Dedicated inference engine worker
        t2_infer = std::thread([&]() {
            try {
                PrepTask in_task;
                while (!abort_flag.load(std::memory_order_relaxed) &&
                       q_prep_to_infer.pop(in_task)) {
                    const auto t0 = Clock::now();
                    RawOutput raw = detector.run_raw(in_task.prep.blob);
                    const auto t1 = Clock::now();

                    InferTask out_task;
                    out_task.frame_id = in_task.frame_id;
                    out_task.name = std::move(in_task.name);
                    out_task.orig_image = std::move(in_task.orig_image);
                    out_task.lb_info = in_task.prep.lb_info;
                    out_task.raw = std::move(raw);
                    out_task.read_ms = in_task.read_ms;
                    out_task.prep_ms = in_task.prep_ms;
                    out_task.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    if (!q_infer_to_post.push(std::move(out_task))) {
                        break;
                    }
                }
            } catch (...) {
                exc_t2 = std::current_exception();
                abort_flag.store(true, std::memory_order_relaxed);
                q_prep_to_infer.close();
            }
            q_infer_to_post.close();
        });

        // Stage 3: Decode + NMS + draw + disk I/O + statistics
        t3_post = std::thread([&]() {
            try {
                InferTask task;
                while (!abort_flag.load(std::memory_order_relaxed) &&
                       q_infer_to_post.pop(task)) {
                    const auto t0 = Clock::now();
                    std::vector<Detection> dets =
                        detector.decode_and_postprocess(task.raw, task.lb_info);
                    const auto t1 = Clock::now();

                    std::array<int, kNumClasses> frame_counts{};
                    frame_counts.fill(0);
                    if (cfg.save_images) {
                        cv::Mat vis = draw_detections(task.orig_image, dets, frame_counts);
                        const fs::path out_path = fs::path(cfg.output_dir) / task.name;
                        if (!cv::imwrite(out_path.string(), vis)) {
                            throw std::runtime_error("Failed to write output image: " +
                                                     out_path.string());
                        }
                    } else {
                        for (const auto& d : dets) {
                            if (d.class_id >= 0 && d.class_id < kNumClasses) {
                                frame_counts[d.class_id] += 1;
                            }
                        }
                    }
                    const auto t2 = Clock::now();

                    if (cfg.verbose) {
                        log_frame_detections(task.name, dets);
                    }

                    for (int c = 0; c < kNumClasses; ++c) {
                        stats.class_counts[c] += frame_counts[c];
                    }
                    stats.processed_frames += 1;

                    sum_read += task.read_ms;
                    sum_prep += task.prep_ms;
                    sum_infer += task.infer_ms;
                    sum_post += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    sum_draw_save += std::chrono::duration<double, std::milli>(t2 - t1).count();
                }
            } catch (...) {
                exc_t3 = std::current_exception();
                abort_flag.store(true, std::memory_order_relaxed);
                q_prep_to_infer.close();
                q_infer_to_post.close();
            }
        });
    }  // All 3 worker threads guaranteed joined before inspecting exceptions/stats.

    if (exc_t1) {
        std::rethrow_exception(exc_t1);
    }
    if (exc_t2) {
        std::rethrow_exception(exc_t2);
    }
    if (exc_t3) {
        std::rethrow_exception(exc_t3);
    }

    const auto wall_end = Clock::now();
    stats.wall_clock_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    stats.dropped_stage1 = q_prep_to_infer.dropped_count();
    stats.dropped_stage2 = q_infer_to_post.dropped_count();
    finalize_averages(stats, sum_read, sum_prep, sum_infer, sum_post, sum_draw_save);
    return stats;
}

}  // namespace steel
