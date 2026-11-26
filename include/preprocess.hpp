#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

#include "types.hpp"

namespace steel {

/**
 * @brief Resize an image preserving aspect ratio and pad to target_w x target_h.
 *        Supports 1-channel grayscale, 3-channel BGR, and 4-channel BGRA inputs.
 */
LetterBoxInfo letterbox(const cv::Mat& src,
                        cv::Mat& dst,
                        int target_w,
                        int target_h,
                        uint8_t pad_val = 114);

/**
 * @brief Convert CV_8UC3 HWC image to normalized float32 CHW planar buffer [0, 1].
 *        Reuses out_blob storage to avoid repeated heap allocations.
 */
void hwc_to_chw_normalize(const cv::Mat& bgr_img,
                          std::vector<float>& out_blob,
                          bool swap_rb = true);

}  // namespace steel
