#pragma once

#include <chrono>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "types.hpp"

namespace steel {

/**
 * @brief Strategy interface for steel defect detectors.
 *        Separates preprocess / inference / postprocess so both multi-backend switching
 *        and 3-stage asynchronous pipeline decoupling reuse the exact same contract.
 */
class IDetector {
public:
    IDetector() = default;
    virtual ~IDetector() = default;

    IDetector(const IDetector&) = delete;
    IDetector& operator=(const IDetector&) = delete;
    IDetector(IDetector&&) = delete;
    IDetector& operator=(IDetector&&) = delete;

    virtual PreprocessedData preprocess(const cv::Mat& bgr_image) const = 0;
    virtual RawOutput run_raw(const std::vector<float>& input_blob) = 0;
    virtual std::vector<Detection> decode_and_postprocess(const RawOutput& raw,
                                                          const LetterBoxInfo& lb) const = 0;

    InferResult infer(const cv::Mat& bgr_image) {
        using Clock = std::chrono::steady_clock;

        InferResult result;
        const auto t0 = Clock::now();
        PreprocessedData prep = preprocess(bgr_image);
        const auto t1 = Clock::now();
        RawOutput raw = run_raw(prep.blob);
        const auto t2 = Clock::now();
        result.detections = decode_and_postprocess(raw, prep.lb_info);
        const auto t3 = Clock::now();

        result.timing.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.timing.infer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        result.timing.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return result;
    }

    virtual std::string backend_name() const = 0;
    virtual void print_model_info() const = 0;
};

}  // namespace steel
