#include "ort_yolov5lite_detector.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

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

OrtYoloV5LiteDetector::OrtYoloV5LiteDetector(const DetectorConfig& config)
    : config_(config),
      decoder_(config.input_w, config.input_h),
      env_(ORT_LOGGING_LEVEL_WARNING, "steel_ort_v5lite"),
      session_options_(),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    if (config_.model_path.empty() || !fs::exists(config_.model_path)) {
        throw std::runtime_error("ONNX model file does not exist: " + config_.model_path);
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

    if (output_shape_.size() != 2 && output_shape_.size() != 3) {
        throw std::runtime_error("Expected 2D/3D YOLOv5-Lite output tensor, got " +
                                 format_shape(output_shape_));
    }
}

void OrtYoloV5LiteDetector::print_model_info() const {
    std::cout << "=== Model Info [" << backend_name() << "] ===\n"
              << "  Model   : " << config_.model_path << "\n"
              << "  Threads : " << config_.num_threads << "\n"
              << "  Input   : " << input_name_ << " " << format_shape(input_shape_) << "\n"
              << "  Output  : " << output_name_ << " " << format_shape(output_shape_) << "\n"
              << "  Config  : input=" << config_.input_w << "x" << config_.input_h
              << ", conf=" << config_.conf_threshold
              << ", iou=" << config_.iou_threshold
              << ", swap_rb=" << (config_.swap_rb ? "true" : "false")
              << ", per_cls_thresh=" << (config_.class_conf_thresholds.empty() ? "false" : "true")
              << "\n";
}

PreprocessedData OrtYoloV5LiteDetector::preprocess(const cv::Mat& bgr_image) const {
    cv::Mat lb_img;
    PreprocessedData prep;
    prep.lb_info = letterbox(bgr_image, lb_img, config_.input_w, config_.input_h, 114);
    hwc_to_chw_normalize(lb_img, prep.blob, config_.swap_rb);
    return prep;
}

RawOutput OrtYoloV5LiteDetector::run_raw(const std::vector<float>& input_blob) {
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

std::vector<Detection> OrtYoloV5LiteDetector::decode_and_postprocess(
    const RawOutput& raw,
    const LetterBoxInfo& lb) const {
    return decoder_.decode(raw, lb, config_);
}

}  // namespace steel
