#include "ort_yolov8_detector.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "postprocess.hpp"
#include "preprocess.hpp"

namespace fs = std::filesystem;

namespace steel {

namespace {

std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

}  // namespace

OrtYoloV8Detector::OrtYoloV8Detector(const DetectorConfig& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "steel_ort_yolov8"),
      session_options_(),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    if (config_.model_path.empty() || !fs::exists(config_.model_path)) {
        throw std::runtime_error("ONNX model file does not exist: " + config_.model_path);
    }
    if (config_.input_w <= 0 || config_.input_h <= 0) {
        throw std::invalid_argument("Invalid model input dimensions in DetectorConfig");
    }
    if (config_.num_threads <= 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
    if (!config_.class_conf_thresholds.empty() &&
        config_.class_conf_thresholds.size() != static_cast<size_t>(kNumClasses)) {
        throw std::invalid_argument("class_conf_thresholds size must match kNumClasses (6)");
    }

    session_options_.SetIntraOpNumThreads(config_.num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, config_.model_path.c_str(), session_options_);

    if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 1) {
        throw std::runtime_error("Model must have at least 1 input and 1 output tensor");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name_ptr = session_->GetInputNameAllocated(0, allocator);
    auto out_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    input_name_ = in_name_ptr.get();
    output_name_ = out_name_ptr.get();

    Ort::TypeInfo in_type = session_->GetInputTypeInfo(0);
    input_shape_ = in_type.GetTensorTypeAndShapeInfo().GetShape();

    Ort::TypeInfo out_type = session_->GetOutputTypeInfo(0);
    output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();

    if (input_shape_.size() != 4) {
        throw std::runtime_error("Expected 4D input tensor [N, C, H, W], got " +
                                 format_shape(input_shape_));
    }
    if (input_shape_[1] > 0 && input_shape_[1] != 3) {
        throw std::runtime_error("Expected 3 input channels, got " +
                                 std::to_string(input_shape_[1]));
    }
    if ((input_shape_[2] > 0 && input_shape_[2] != config_.input_h) ||
        (input_shape_[3] > 0 && input_shape_[3] != config_.input_w)) {
        throw std::runtime_error("Model static input HxW " + format_shape(input_shape_) +
                                 " does not match configured " +
                                 std::to_string(config_.input_h) + "x" +
                                 std::to_string(config_.input_w));
    }

    if (output_shape_.size() != 3) {
        throw std::runtime_error("Expected 3D YOLOv8 output tensor [1, C, N], got " +
                                 format_shape(output_shape_));
    }
    if (output_shape_[1] > 0 && output_shape_[1] < 5) {
        throw std::runtime_error("Invalid YOLOv8 output channel dimension: " +
                                 std::to_string(output_shape_[1]));
    }
}

void OrtYoloV8Detector::print_model_info() const {
    std::cout << "=== Model Info [" << backend_name() << "] ===\n"
              << "  Model   : " << config_.model_path << "\n"
              << "  Threads : " << config_.num_threads << "\n"
              << "  Input   : " << input_name_ << " " << format_shape(input_shape_) << "\n"
              << "  Output  : " << output_name_ << " " << format_shape(output_shape_) << "\n"
              << "  Config  : input=" << config_.input_w << "x" << config_.input_h
              << ", conf=" << config_.conf_threshold
              << ", iou=" << config_.iou_threshold
              << ", swap_rb=" << (config_.swap_rb ? "true" : "false") << "\n";
}

PreprocessedData OrtYoloV8Detector::preprocess(const cv::Mat& bgr_image) const {
    cv::Mat lb_img;
    PreprocessedData prep;
    prep.lb_info = letterbox(bgr_image, lb_img, config_.input_w, config_.input_h, 114);
    hwc_to_chw_normalize(lb_img, prep.blob, config_.swap_rb);
    return prep;
}

RawOutput OrtYoloV8Detector::run_raw(const std::vector<float>& input_blob) {
    const size_t expected_elems =
        static_cast<size_t>(3) * static_cast<size_t>(config_.input_h) * static_cast<size_t>(config_.input_w);
    if (input_blob.size() != expected_elems) {
        throw std::invalid_argument("Input blob size mismatch: expected " +
                                    std::to_string(expected_elems) + ", got " +
                                    std::to_string(input_blob.size()));
    }

    std::array<int64_t, 4> run_shape = {1, 3, config_.input_h, config_.input_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(input_blob.data()),
        input_blob.size(),
        run_shape.data(),
        run_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("ORT session.Run returned empty or non-tensor output");
    }

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    RawOutput raw;
    raw.shape = info.GetShape();
    const size_t out_count = info.GetElementCount();
    const float* out_ptr = outputs[0].GetTensorData<float>();
    raw.data.assign(out_ptr, out_ptr + out_count);
    return raw;
}

std::vector<Detection> OrtYoloV8Detector::decode_and_postprocess(const RawOutput& raw,
                                                                 const LetterBoxInfo& lb) const {
    if (raw.shape.size() != 3 || raw.shape[0] != 1) {
        throw std::runtime_error("Expected [1, C, N] raw output shape, got " +
                                 format_shape(raw.shape));
    }

    const int channels = static_cast<int>(raw.shape[1]);
    const int num_anchors = static_cast<int>(raw.shape[2]);
    if (channels < 5 || num_anchors <= 0) {
        throw std::runtime_error("Invalid YOLOv8 output dimensions: " + format_shape(raw.shape));
    }

    const size_t expected_size = static_cast<size_t>(channels) * static_cast<size_t>(num_anchors);
    if (raw.data.size() != expected_size) {
        throw std::runtime_error("Raw output buffer size mismatch");
    }

    const int num_classes = channels - 4;
    const bool use_per_class_thresh =
        config_.class_conf_thresholds.size() == static_cast<size_t>(num_classes);

    const float* data = raw.data.data();
    std::vector<Detection> candidates;
    candidates.reserve(64);

    // Memory layout is [1, 4 + num_classes, num_anchors] in row-major order (stride = num_anchors).
    for (int i = 0; i < num_anchors; ++i) {
        int best_cls = 0;
        float best_score = data[4 * num_anchors + i];
        for (int c = 1; c < num_classes; ++c) {
            const float score = data[(4 + c) * num_anchors + i];
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }

        const float thresh = use_per_class_thresh
                                 ? config_.class_conf_thresholds[best_cls]
                                 : config_.conf_threshold;
        if (best_score < thresh) {
            continue;
        }

        const float cx = data[0 * num_anchors + i];
        const float cy = data[1 * num_anchors + i];
        const float w = data[2 * num_anchors + i];
        const float h = data[3 * num_anchors + i];

        Detection det;
        det.x1 = cx - w * 0.5f;
        det.y1 = cy - h * 0.5f;
        det.x2 = cx + w * 0.5f;
        det.y2 = cy + h * 0.5f;
        det.score = best_score;
        det.class_id = best_cls;
        candidates.push_back(det);
    }

    std::vector<Detection> kept = nms(std::move(candidates), config_.iou_threshold, config_.class_agnostic_nms);
    restore_coordinates(kept, lb);
    return kept;
}

}  // namespace steel
