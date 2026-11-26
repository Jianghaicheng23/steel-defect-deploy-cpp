#include "frame_source.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace steel {

ImageListSource::ImageListSource(std::vector<fs::path> image_paths, int max_frames)
    : paths_(std::move(image_paths)), max_frames_(max_frames) {}

bool ImageListSource::next(FramePacket& out) {
    using Clock = std::chrono::steady_clock;

    while (cursor_ < paths_.size()) {
        if (max_frames_ > 0 && emitted_ >= max_frames_) {
            return false;
        }
        const fs::path& img_path = paths_[cursor_++];
        const auto t0 = Clock::now();
        cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        const auto t1 = Clock::now();

        if (img.empty()) {
            std::cerr << "Warning: failed to read image, skipping: " << img_path << "\n";
            continue;
        }

        out.frame_id = emitted_++;
        out.name = img_path.filename().string();
        out.image = std::move(img);
        out.read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }
    return false;
}

VideoCaptureSource::VideoCaptureSource(const std::string& video_path, int max_frames)
    : cap_(video_path), max_frames_(max_frames) {
    if (!cap_.isOpened()) {
        throw std::runtime_error("Failed to open video file: " + video_path);
    }
}

VideoCaptureSource::VideoCaptureSource(int camera_index, int max_frames)
    : cap_(camera_index), max_frames_(max_frames) {
    if (!cap_.isOpened()) {
        throw std::runtime_error("Failed to open camera index: " + std::to_string(camera_index));
    }
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
}

VideoCaptureSource::~VideoCaptureSource() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoCaptureSource::next(FramePacket& out) {
    using Clock = std::chrono::steady_clock;

    if (max_frames_ > 0 && emitted_ >= max_frames_) {
        return false;
    }

    cv::Mat frame;
    const auto t0 = Clock::now();
    const bool ok = cap_.read(frame);
    const auto t1 = Clock::now();

    if (!ok || frame.empty()) {
        return false;
    }

    char name_buf[32];
    std::snprintf(name_buf, sizeof(name_buf), "frame_%06d.jpg", emitted_);

    out.frame_id = emitted_++;
    out.name = name_buf;
    out.image = std::move(frame);
    out.read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

}  // namespace steel
