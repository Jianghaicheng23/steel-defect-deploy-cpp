#include "mnn_yolov5lite_detector.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cstring>
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

MnnYoloV5LiteDetector::MnnYoloV5LiteDetector(const DetectorConfig& config)
    : config_(config),
      decoder_(config.input_w, config.input_h) {
    if (config_.model_path.empty() || !fs::exists(config_.model_path)) {
        throw std::runtime_error("MNN model file does not exist: " + config_.model_path);
    }
    if (config_.num_threads <= 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
    if (!config_.class_conf_thresholds.empty() &&
        config_.class_conf_thresholds.size() != static_cast<size_t>(kNumClasses)) {
        throw std::invalid_argument("class_conf_thresholds size must match kNumClasses (6)");
    }

    interpreter_ = MNN::Interpreter::createFromFile(config_.model_path.c_str());
    if (!interpreter_) {
        throw std::runtime_error("Failed to load MNN model: " + config_.model_path);
    }

    MNN::BackendConfig backend_cfg;
    backend_cfg.precision = MNN::BackendConfig::Precision_Low;

    MNN::ScheduleConfig sched_cfg;
    sched_cfg.type = MNN_FORWARD_CPU;
    sched_cfg.numThread = config_.num_threads;
    sched_cfg.backendConfig = &backend_cfg;

    session_ = interpreter_->createSession(sched_cfg);
    if (!session_) {
        MNN::Interpreter::destroy(interpreter_);
        interpreter_ = nullptr;
        throw std::runtime_error("Failed to create MNN session");
    }

    input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor_) {
        interpreter_->releaseSession(session_);
        MNN::Interpreter::destroy(interpreter_);
        session_ = nullptr;
        interpreter_ = nullptr;
        throw std::runtime_error("Failed to get MNN session input tensor");
    }

    interpreter_->resizeTensor(input_tensor_, {1, 3, config_.input_h, config_.input_w});
    interpreter_->resizeSession(session_);

    output_tensor_ = interpreter_->getSessionOutput(session_, nullptr);
    if (!output_tensor_) {
        interpreter_->releaseSession(session_);
        MNN::Interpreter::destroy(interpreter_);
        session_ = nullptr;
        interpreter_ = nullptr;
        throw std::runtime_error("Failed to get MNN session output tensor");
    }

    host_input_ = new MNN::Tensor(input_tensor_, MNN::Tensor::CAFFE);
    host_output_ = new MNN::Tensor(output_tensor_, MNN::Tensor::CAFFE);

    for (int d : input_tensor_->shape()) {
        input_shape_.push_back(static_cast<int64_t>(d));
    }
    for (int d : output_tensor_->shape()) {
        output_shape_.push_back(static_cast<int64_t>(d));
    }
}

MnnYoloV5LiteDetector::~MnnYoloV5LiteDetector() {
    if (host_input_) {
        MNN::Tensor::destroy(host_input_);
        host_input_ = nullptr;
    }
    if (host_output_) {
        MNN::Tensor::destroy(host_output_);
        host_output_ = nullptr;
    }
    if (interpreter_ && session_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    if (interpreter_) {
        MNN::Interpreter::destroy(interpreter_);
        interpreter_ = nullptr;
    }
    }


void MnnYoloV5LiteDetector::print_model_info() const {
    std::cout << "=== Model Info [" << backend_name() << "] ===\n"
              << "  Model   : " << config_.model_path << "\n"
              << "  Threads : " << config_.num_threads << "\n"
              << "  Input   : " << format_shape(input_shape_) << "\n"
              << "  Output  : " << format_shape(output_shape_) << "\n"
              << "  Config  : input=" << config_.input_w << "x" << config_.input_h
              << ", conf=" << config_.conf_threshold
              << ", iou=" << config_.iou_threshold
              << ", swap_rb=" << (config_.swap_rb ? "true" : "false")
              << ", per_cls_thresh=" << (config_.class_conf_thresholds.empty() ? "false" : "true")
              << "\n";
}

PreprocessedData MnnYoloV5LiteDetector::preprocess(const cv::Mat& bgr_image) const {
    cv::Mat lb_img;
    PreprocessedData prep;
    prep.lb_info = letterbox(bgr_image, lb_img, config_.input_w, config_.input_h, 114);
    hwc_to_chw_normalize(lb_img, prep.blob, config_.swap_rb);
    return prep;
}

RawOutput MnnYoloV5LiteDetector::run_raw(const std::vector<float>& input_blob) {
    const size_t expected_elems =
        static_cast<size_t>(3) * static_cast<size_t>(config_.input_h) * static_cast<size_t>(config_.input_w);
    if (input_blob.size() != expected_elems) {
        throw std::invalid_argument("Input blob size mismatch: expected " +
                                    std::to_string(expected_elems) + ", got " +
                                    std::to_string(input_blob.size()));
    }

    float* host_in_ptr = host_input_->host<float>();
    if (!host_in_ptr) {
        throw std::runtime_error("MNN host input buffer is null");
    }
    std::memcpy(host_in_ptr, input_blob.data(), input_blob.size() * sizeof(float));

    input_tensor_->copyFromHostTensor(host_input_);
    const MNN::ErrorCode err = interpreter_->runSession(session_);
    if (err != MNN::NO_ERROR) {
        throw std::runtime_error("MNN runSession failed with error code: " +
                                 std::to_string(static_cast<int>(err)));
    }
    output_tensor_->copyToHostTensor(host_output_);

    const int out_elems = host_output_->elementSize();
    const float* host_out_ptr = host_output_->host<float>();
    if (!host_out_ptr || out_elems <= 0) {
        throw std::runtime_error("MNN host output buffer is invalid");
    }

    RawOutput raw;
    raw.shape = output_shape_;
    raw.data.assign(host_out_ptr, host_out_ptr + out_elems);
    return raw;
}

std::vector<Detection> MnnYoloV5LiteDetector::decode_and_postprocess(
    const RawOutput& raw,
    const LetterBoxInfo& lb) const {
    return decoder_.decode(raw, lb, config_);
}

}  // namespace steel
