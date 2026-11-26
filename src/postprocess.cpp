#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace steel {

float compute_iou(const Detection& a, const Detection& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);

    const float inter_w = std::max(0.0f, xx2 - xx1);
    const float inter_h = std::max(0.0f, yy2 - yy1);
    const float inter_area = inter_w * inter_h;
    if (inter_area <= 0.0f) {
        return 0.0f;
    }

    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float union_area = area_a + area_b - inter_area;
    if (union_area <= 1e-6f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

std::vector<Detection> nms(std::vector<Detection> candidates,
                           float iou_threshold,
                           bool class_agnostic) {
    if (candidates.empty()) {
        return {};
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& lhs, const Detection& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<uint8_t> suppressed(candidates.size(), 0);
    std::vector<Detection> kept;
    kept.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        const Detection& top = candidates[i];
        kept.push_back(top);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (!class_agnostic && top.class_id != candidates[j].class_id) {
                continue;
            }
            if (compute_iou(top, candidates[j]) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }

    return kept;
}

void restore_coordinates(std::vector<Detection>& dets, const LetterBoxInfo& lb) {
    if (lb.scale <= 0.0f || lb.orig_w <= 0 || lb.orig_h <= 0) {
        return;
    }

    const float max_x = static_cast<float>(lb.orig_w - 1);
    const float max_y = static_cast<float>(lb.orig_h - 1);
    const float inv_scale = 1.0f / lb.scale;
    const float pad_x = static_cast<float>(lb.pad_x);
    const float pad_y = static_cast<float>(lb.pad_y);

    for (auto& det : dets) {
        det.x1 = std::clamp((det.x1 - pad_x) * inv_scale, 0.0f, max_x);
        det.y1 = std::clamp((det.y1 - pad_y) * inv_scale, 0.0f, max_y);
        det.x2 = std::clamp((det.x2 - pad_x) * inv_scale, 0.0f, max_x);
        det.y2 = std::clamp((det.y2 - pad_y) * inv_scale, 0.0f, max_y);
        if (det.x2 < det.x1) {
            std::swap(det.x1, det.x2);
        }
        if (det.y2 < det.y1) {
            std::swap(det.y1, det.y2);
        }
    }
}

cv::Mat draw_detections(const cv::Mat& image,
                        const std::vector<Detection>& dets,
                        std::array<int, kNumClasses>& out_counts) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty in draw_detections()");
    }

    out_counts.fill(0);

    cv::Mat canvas;
    if (image.channels() == 1) {
        cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, canvas, cv::COLOR_BGRA2BGR);
    } else {
        canvas = image.clone();
    }

    const int min_dim = std::min(canvas.cols, canvas.rows);
    const double font_scale = std::clamp(static_cast<double>(min_dim) / 500.0, 0.35, 0.65);
    const int box_thickness = std::max(1, static_cast<int>(std::round(static_cast<double>(min_dim) / 160.0)));
    const int font_thickness = 1;

    int total_count = 0;
    for (const auto& det : dets) {
        const int cls_id = (det.class_id >= 0 && det.class_id < kNumClasses) ? det.class_id : -1;
        const std::string cls_name = (cls_id >= 0) ? kClassNames[cls_id] : "unknown";
        const cv::Scalar color = (cls_id >= 0) ? kClassColors[cls_id] : cv::Scalar(200, 200, 200);

        if (cls_id >= 0) {
            out_counts[cls_id] += 1;
        }
        total_count += 1;

        const int x1 = std::clamp(static_cast<int>(std::round(det.x1)), 0, canvas.cols - 1);
        const int y1 = std::clamp(static_cast<int>(std::round(det.y1)), 0, canvas.rows - 1);
        const int x2 = std::clamp(static_cast<int>(std::round(det.x2)), 0, canvas.cols - 1);
        const int y2 = std::clamp(static_cast<int>(std::round(det.y2)), 0, canvas.rows - 1);

        cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thickness);

        char label_buf[64];
        std::snprintf(label_buf, sizeof(label_buf), "%s %.2f", cls_name.c_str(), det.score);
        const std::string label(label_buf);

        int baseline = 0;
        const cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        const int pad = 2;
        const int box_h = text_sz.height + baseline + pad * 2;
        const int box_w = text_sz.width + pad * 2;

        int bg_x1 = std::min(x1, std::max(0, canvas.cols - box_w));
        int bg_x2 = std::min(canvas.cols - 1, bg_x1 + box_w);
        int bg_y1 = y1 - box_h;
        int bg_y2 = y1;
        if (bg_y1 < 0) {
            // Keep label inside frame when box is near top edge (important for 200x200 images)
            bg_y1 = y1;
            bg_y2 = std::min(canvas.rows - 1, y1 + box_h);
        }

        cv::rectangle(canvas, cv::Point(bg_x1, bg_y1), cv::Point(bg_x2, bg_y2), color, cv::FILLED);
        const int text_y = std::max(text_sz.height + pad, bg_y2 - baseline - pad);
        cv::putText(canvas, label, cv::Point(bg_x1 + pad, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                    font_thickness, cv::LINE_AA);
    }

    char header_buf[64];
    std::snprintf(header_buf, sizeof(header_buf), "Detect: %d objects", total_count);
    const double header_scale = std::clamp(static_cast<double>(min_dim) / 450.0, 0.4, 0.7);
    cv::putText(canvas, header_buf, cv::Point(6, 16),
                cv::FONT_HERSHEY_SIMPLEX, header_scale, cv::Scalar(0, 255, 255),
                1, cv::LINE_AA);

    return canvas;
}

}  // namespace steel
