#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"

namespace steel {

class OrtYoloV8Detector final : public IDetector {
public:
    explicit OrtYoloV8Detector(const DetectorConfig& config);
    ~OrtYoloV8Detector() override = default;

    PreprocessedData preprocess(const cv::Mat& bgr_image) const override;
    RawOutput run_raw(const std::vector<float>& input_blob) override;
    std::vector<Detection> decode_and_postprocess(const RawOutput& raw,
                                                  const LetterBoxInfo& lb) const override;

    std::string backend_name() const override { return "ONNXRuntime-CPU (YOLOv8)"; }
    void print_model_info() const override;

private:
    DetectorConfig config_;
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
