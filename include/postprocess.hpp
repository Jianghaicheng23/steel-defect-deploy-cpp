#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <vector>

#include "types.hpp"

namespace steel {

/**
 * @brief Compute Intersection-over-Union (IoU) of two xyxy boxes.
 */
float compute_iou(const Detection& a, const Detection& b);

/**
 * @brief Pure C++ Non-Maximum Suppression supporting per-class or class-agnostic mode.
 */
std::vector<Detection> nms(std::vector<Detection> candidates,
                           float iou_threshold,
                           bool class_agnostic = false);

/**
 * @brief Map box coordinates from letterbox target space back to original image space.
 */
void restore_coordinates(std::vector<Detection>& dets, const LetterBoxInfo& lb);

/**
 * @brief Draw detection boxes, labels, and summary text onto a copy of image,
 *        and populate per-class detection counts.
 */
cv::Mat draw_detections(const cv::Mat& image,
                        const std::vector<Detection>& dets,
                        std::array<int, kNumClasses>& out_counts);

}  // namespace steel
