#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace steel {

LetterBoxInfo letterbox(const cv::Mat& src,
                        cv::Mat& dst,
                        int target_w,
                        int target_h,
                        uint8_t pad_val) {
    if (src.empty()) {
        throw std::invalid_argument("Input image is empty in letterbox()");
    }
    if (target_w <= 0 || target_h <= 0) {
        throw std::invalid_argument("Target dimensions must be positive in letterbox()");
    }

    cv::Mat bgr;
    if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else if (src.channels() == 3) {
        bgr = src;
    } else {
        throw std::invalid_argument("Unsupported channel count in letterbox()");
    }

    if (bgr.depth() != CV_8U) {
        bgr.convertTo(bgr, CV_8UC3);
    }

    const int orig_w = bgr.cols;
    const int orig_h = bgr.rows;

    const float scale = std::min(static_cast<float>(target_w) / static_cast<float>(orig_w),
                                 static_cast<float>(target_h) / static_cast<float>(orig_h));
    const int new_w = std::min(target_w, std::max(1, static_cast<int>(std::round(orig_w * scale))));
    const int new_h = std::min(target_h, std::max(1, static_cast<int>(std::round(orig_h * scale))));

    const int pad_x = (target_w - new_w) / 2;
    const int pad_y = (target_h - new_h) / 2;
    const int pad_right = target_w - new_w - pad_x;
    const int pad_bottom = target_h - new_h - pad_y;

    cv::Mat resized;
    if (new_w != orig_w || new_h != orig_h) {
        cv::resize(bgr, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = bgr;
    }

    if (pad_x == 0 && pad_right == 0 && pad_y == 0 && pad_bottom == 0) {
        if (resized.isContinuous()) {
            dst = resized;
        } else {
            resized.copyTo(dst);
        }
    } else {
        cv::copyMakeBorder(resized, dst,
                           pad_y, pad_bottom, pad_x, pad_right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(pad_val, pad_val, pad_val));
    }

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    info.orig_w = orig_w;
    info.orig_h = orig_h;
    info.target_w = target_w;
    info.target_h = target_h;
    return info;
}

void hwc_to_chw_normalize(const cv::Mat& bgr_img,
                          std::vector<float>& out_blob,
                          bool swap_rb) {
    if (bgr_img.empty() || bgr_img.type() != CV_8UC3)//8U即是uint8,C3即就是3个channel
    {
        throw std::invalid_argument("Expected non-empty CV_8UC3 image in hwc_to_chw_normalize()");
    }

    const int height = bgr_img.rows;
    const int width = bgr_img.cols;
    const size_t area = static_cast<size_t>(height) * static_cast<size_t>(width);

    out_blob.resize(3 * area);//三通道，相当于准备3个矩阵

    constexpr float kInv255 = 1.0f / 255.0f;
    float* p_c0 = out_blob.data();
    float* p_c1 = out_blob.data() + area;
    float* p_c2 = out_blob.data() + 2 * area;

    if (swap_rb) {//BGR-->RGB
        for (int h = 0; h < height; ++h) {
            const uint8_t* row = bgr_img.ptr<uint8_t>(h);
            for (int w = 0; w < width; ++w) {
                const int idx = w * 3;
                *p_c0++ = static_cast<float>(row[idx + 2]) * kInv255;  // R把像素从0-255转变为0-1
                *p_c1++ = static_cast<float>(row[idx + 1]) * kInv255;  // G把像素从0-255转变为0-1
                *p_c2++ = static_cast<float>(row[idx + 0]) * kInv255;  // B把像素从0-255转变为0-1
            }
        }
    } else {//不更换通道
        for (int h = 0; h < height; ++h) {
            const uint8_t* row = bgr_img.ptr<uint8_t>(h);
            for (int w = 0; w < width; ++w) {
                const int idx = w * 3;
                *p_c0++ = static_cast<float>(row[idx + 0]) * kInv255;  // B
                *p_c1++ = static_cast<float>(row[idx + 1]) * kInv255;  // G
                *p_c2++ = static_cast<float>(row[idx + 2]) * kInv255;  // R
            }
        }
    }
}

}  // namespace steel
