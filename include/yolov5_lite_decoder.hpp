#pragma once

#include <array>
#include <vector>

#include "types.hpp"

namespace steel {

/**
 * @brief Anchor-based decoder for YOLOv5-Lite exported with Detect::cat_forward
 *        (output shape [1, total_anchors, 5 + num_classes], sigmoid already applied).
 */
class YoloV5LiteDecoder {
public:
    explicit YoloV5LiteDecoder(int input_w = 320, int input_h = 320);

    int total_anchors() const noexcept { return total_anchors_; }

    /**
     * @brief Per-class confidence thresholds matching Pre_Project/my_detect_mnn.py.
     */
    static std::vector<float> default_class_thresholds();

    std::vector<Detection> decode(const RawOutput& raw,
                                  const LetterBoxInfo& lb,
                                  const DetectorConfig& cfg) const;

private:
    struct LayerMeta {
        float stride = 8.0f;
        int ny = 0;
        int nx = 0;
        std::array<std::array<float, 2>, 3> anchors{};
    };

    int input_w_ = 320;
    int input_h_ = 320;
    int total_anchors_ = 0;
    std::array<LayerMeta, 3> layers_{};
};

}  // namespace steel
