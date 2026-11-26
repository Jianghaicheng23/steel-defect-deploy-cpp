#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

namespace steel {

struct FramePacket {
    int frame_id = 0;
    std::string name;
    cv::Mat image;
    double read_ms = 0.0;
};

class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual bool next(FramePacket& out) = 0;
};

class ImageListSource final : public IFrameSource {
public:
    explicit ImageListSource(std::vector<std::filesystem::path> image_paths,
                             int max_frames = 0);

    bool next(FramePacket& out) override;

private:
    std::vector<std::filesystem::path> paths_;
    int max_frames_ = 0;
    size_t cursor_ = 0;
    int emitted_ = 0;
};

class VideoCaptureSource final : public IFrameSource {
public:
    explicit VideoCaptureSource(const std::string& video_path, int max_frames = 0);
    explicit VideoCaptureSource(int camera_index, int max_frames = 0);
    ~VideoCaptureSource() override;

    VideoCaptureSource(const VideoCaptureSource&) = delete;
    VideoCaptureSource& operator=(const VideoCaptureSource&) = delete;

    bool next(FramePacket& out) override;

private:
    cv::VideoCapture cap_;
    int max_frames_ = 0;
    int emitted_ = 0;
};

}  // namespace steel
