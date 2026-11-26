#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"
#include "yolov5_lite_decoder.hpp"

namespace steel {

/**
 * @brief ONNX Runtime backend for YOLOv5-Lite (320x320 anchor-based detection).
 */
class OrtYoloV5LiteDetector final : public IDetector {
public:
    explicit OrtYoloV5LiteDetector(const DetectorConfig& config);
    ~OrtYoloV5LiteDetector() override = default;

    PreprocessedData preprocess(const cv::Mat& bgr_image) const override;
    RawOutput run_raw(const std::vector<float>& input_blob) override;
    std::vector<Detection> decode_and_postprocess(const RawOutput& raw,
                                                  const LetterBoxInfo& lb) const override;

    std::string backend_name() const override { return "ONNXRuntime-CPU (YOLOv5-Lite)"; }
    void print_model_info() const override;

private:
    DetectorConfig config_;
    YoloV5LiteDecoder decoder_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo mem_info_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

}  // namespace steel
