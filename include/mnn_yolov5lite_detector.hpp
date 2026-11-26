#pragma once

#include <string>
#include <vector>

#include "detector.hpp"
#include "yolov5_lite_decoder.hpp"

namespace MNN {
class Interpreter;
class Session;
class Tensor;
}  // namespace MNN

namespace steel {

/**
 * @brief MNN CPU(FP32/FP16) backend for YOLOv5-Lite (.mnn model), compiled when -DENABLE_MNN=ON.
 */
class MnnYoloV5LiteDetector final : public IDetector {
public:
    explicit MnnYoloV5LiteDetector(const DetectorConfig& config);
    ~MnnYoloV5LiteDetector() override;

    PreprocessedData preprocess(const cv::Mat& bgr_image) const override;
    RawOutput run_raw(const std::vector<float>& input_blob) override;
    std::vector<Detection> decode_and_postprocess(const RawOutput& raw,
                                                  const LetterBoxInfo& lb) const override;

    std::string backend_name() const override { return "MNN-CPU (YOLOv5-Lite)"; }
    void print_model_info() const override;

private:
    DetectorConfig config_;
    YoloV5LiteDecoder decoder_;
    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    MNN::Tensor* output_tensor_ = nullptr;
    MNN::Tensor* host_input_ = nullptr;
    MNN::Tensor* host_output_ = nullptr;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

}  // namespace steel
