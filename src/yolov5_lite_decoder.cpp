#include "yolov5_lite_decoder.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "postprocess.hpp"

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

YoloV5LiteDecoder::YoloV5LiteDecoder(int input_w, int input_h)
    : input_w_(input_w), input_h_(input_h) {
    if (input_w_ <= 0 || input_h_ <= 0 || (input_w_ % 32 != 0) || (input_h_ % 32 != 0)) {
        throw std::invalid_argument(
            "YOLOv5-Lite input dimensions must be positive multiples of 32");
    }

    constexpr std::array<float, 3> kStrides = {8.0f, 16.0f, 32.0f};
    constexpr std::array<std::array<std::array<float, 2>, 3>, 3> kAnchors = {{
        {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}},
        {{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}},
        {{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}}
    }};

    total_anchors_ = 0;
    for (size_t l = 0; l < layers_.size(); ++l) {
        const int stride_i = static_cast<int>(kStrides[l]);
        layers_[l].stride = kStrides[l];
        layers_[l].ny = input_h_ / stride_i;
        layers_[l].nx = input_w_ / stride_i;
        layers_[l].anchors = kAnchors[l];
        total_anchors_ += 3 * layers_[l].ny * layers_[l].nx;
    }
}

std::vector<float> YoloV5LiteDecoder::default_class_thresholds() {
    // Matches CLASS_CONF_THRESH in Pre_Project/my_detect_mnn.py:
    // crazing=0.20, inclusion=0.60, patches=0.85, pitted_surface=0.45,
    // rolled-in_scale=0.35, scratches=0.80
    return {0.20f, 0.60f, 0.85f, 0.45f, 0.35f, 0.80f};
}

std::vector<Detection> YoloV5LiteDecoder::decode(const RawOutput& raw,
                                                 const LetterBoxInfo& lb,
                                                 const DetectorConfig& cfg) const {
    int num_pred = 0;
    int num_out = 0;

    if (raw.shape.size() == 3) {
        if (raw.shape[0] != 1) {
            throw std::runtime_error("Expected batch=1 for YOLOv5-Lite output, got " +
                                     format_shape(raw.shape));
        }
        num_pred = static_cast<int>(raw.shape[1]);
        num_out = static_cast<int>(raw.shape[2]);
    } else if (raw.shape.size() == 2) {
        num_pred = static_cast<int>(raw.shape[0]);
        num_out = static_cast<int>(raw.shape[1]);
    } else {
        throw std::runtime_error("Expected 2D or 3D YOLOv5-Lite output tensor, got " +
                                 format_shape(raw.shape));
    }

    if (num_pred != total_anchors_ || num_out < 6) {
        throw std::runtime_error("YOLOv5-Lite output shape mismatch: expected [" +
                                 std::to_string(total_anchors_) + ", >=6], got " +
                                 format_shape(raw.shape));
    }

    const size_t expected_elems = static_cast<size_t>(num_pred) * static_cast<size_t>(num_out);
    if (raw.data.size() != expected_elems) {
        throw std::runtime_error("YOLOv5-Lite raw output buffer size mismatch");
    }

    const int num_classes = num_out - 5;
    const bool use_per_class_thresh =
        cfg.class_conf_thresholds.size() == static_cast<size_t>(num_classes);

    const float* data = raw.data.data();
    std::vector<Detection> candidates;
    candidates.reserve(64);

    int row_idx = 0;
    for (const auto& layer : layers_) {
        const float stride = layer.stride;
        for (int a = 0; a < 3; ++a) {
            const float anc_w = layer.anchors[a][0];
            const float anc_h = layer.anchors[a][1];
            for (int gy = 0; gy < layer.ny; ++gy) {
                for (int gx = 0; gx < layer.nx; ++gx) {
                    const float* p = data + static_cast<size_t>(row_idx) * num_out;
                    ++row_idx;

                    const float obj = p[4];
                    int best_cls = 0;
                    float best_cls_prob = p[5];
                    for (int c = 1; c < num_classes; ++c) {
                        const float prob = p[5 + c];
                        if (prob > best_cls_prob) {
                            best_cls_prob = prob;
                            best_cls = c;
                        }
                    }

                    const float best_score = obj * best_cls_prob;
                    const float thresh = use_per_class_thresh
                                             ? cfg.class_conf_thresholds[best_cls]
                                             : cfg.conf_threshold;
                    if (best_score <= thresh) {
                        continue;
                    }

                    const float cx = (p[0] * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
                    const float cy = (p[1] * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
                    const float tw = p[2] * 2.0f;
                    const float th = p[3] * 2.0f;
                    const float w = (tw * tw) * anc_w;
                    const float h = (th * th) * anc_h;

                    Detection det;
                    det.x1 = cx - w * 0.5f;
                    det.y1 = cy - h * 0.5f;
                    det.x2 = cx + w * 0.5f;
                    det.y2 = cy + h * 0.5f;
                    det.score = best_score;
                    det.class_id = best_cls;
                    candidates.push_back(det);
                }
            }
        }
    }

    restore_coordinates(candidates, lb);
    return nms(std::move(candidates), cfg.iou_threshold, cfg.class_agnostic_nms);
}

}  // namespace steel
