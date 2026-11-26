#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace steel {

inline constexpr int kNumClasses = 6;

inline const std::array<std::string, kNumClasses> kClassNames = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
};

inline const std::array<cv::Scalar, kNumClasses> kClassColors = {
    cv::Scalar(0, 255, 0),      // crazing: green
    cv::Scalar(255, 128, 0),    // inclusion: blue-cyan
    cv::Scalar(0, 0, 255),      // patches: red
    cv::Scalar(0, 220, 255),    // pitted_surface: yellow
    cv::Scalar(255, 0, 255),    // rolled-in_scale: magenta
    cv::Scalar(255, 255, 0)     // scratches: cyan
};

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
};

struct LetterBoxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int orig_w = 0;
    int orig_h = 0;
    int target_w = 0;
    int target_h = 0;
};

struct PreprocessedData {
    std::vector<float> blob;
    LetterBoxInfo lb_info;
};

struct RawOutput {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct StageTiming {
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;

    double total_ms() const {
        return preprocess_ms + infer_ms + postprocess_ms;
    }
};

struct InferResult {
    std::vector<Detection> detections;
    StageTiming timing;
};

struct DetectorConfig {
    std::string model_path;
    int input_w = 640;
    int input_h = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int num_threads = 4;
    bool swap_rb = true;              // true for YOLOv8 (RGB); false for YOLOv5-Lite MNN (BGR)
    bool class_agnostic_nms = false;  // false: per-class NMS; true: global NMS
    std::vector<float> class_conf_thresholds;
};

}  // namespace steel
