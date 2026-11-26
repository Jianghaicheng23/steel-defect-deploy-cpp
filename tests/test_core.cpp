#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bounded_safe_queue.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "types.hpp"
#include "yolov5_lite_decoder.hpp"

#define TEST_CHECK(cond)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream oss;                                            \
            oss << "Assertion failed at " << __FILE__ << ":" << __LINE__       \
                << " (" #cond ")";                                             \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (false)

namespace {

void test_bounded_queue_fifo_and_block() {
    steel::BoundedSafeQueue<int> q(4, steel::OverflowPolicy::kBlock);
    constexpr int kCount = 200;
    std::vector<int> consumed;
    consumed.reserve(kCount);

    std::thread producer([&]() {
        for (int i = 0; i < kCount; ++i) {
            TEST_CHECK(q.push(i));
        }
        q.close();
    });

    std::thread consumer([&]() {
        int val = -1;
        while (q.pop(val)) {
            consumed.push_back(val);
        }
    });

    producer.join();
    consumer.join();

    TEST_CHECK(consumed.size() == static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        TEST_CHECK(consumed[static_cast<size_t>(i)] == i);
    }
    TEST_CHECK(q.dropped_count() == 0);
    TEST_CHECK(q.is_closed());
}

void test_bounded_queue_drop_oldest() {
    steel::BoundedSafeQueue<int> q(4, steel::OverflowPolicy::kDropOldest);
    for (int i = 0; i < 50; ++i) {
        TEST_CHECK(q.push(i));
    }
    q.close();

    TEST_CHECK(q.dropped_count() == 46);
    TEST_CHECK(q.size() == 4);

    std::vector<int> out;
    int val = -1;
    while (q.pop(val)) {
        out.push_back(val);
    }
    TEST_CHECK(out == (std::vector<int>{46, 47, 48, 49}));
}

void test_bounded_queue_close_unblocks_waiters() {
    // Empty queue pop waiter
    steel::BoundedSafeQueue<int> empty_q(2, steel::OverflowPolicy::kBlock);
    std::atomic<bool> pop_result{true};
    std::thread t_pop([&]() {
        int v = 0;
        pop_result.store(empty_q.pop(v));
    });

    // Full queue push waiter
    steel::BoundedSafeQueue<int> full_q(1, steel::OverflowPolicy::kBlock);
    TEST_CHECK(full_q.push(1));
    std::atomic<bool> push_result{true};
    std::thread t_push([&]() {
        push_result.store(full_q.push(2));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    empty_q.close();
    full_q.close();

    t_pop.join();
    t_push.join();

    TEST_CHECK(!pop_result.load());
    TEST_CHECK(!push_result.load());
}

void test_letterbox_and_restore_roundtrip() {
    cv::Mat src(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat dst;
    steel::LetterBoxInfo lb = steel::letterbox(src, dst, 320, 320, 114);

    TEST_CHECK(dst.cols == 320 && dst.rows == 320);
    TEST_CHECK(std::fabs(lb.scale - 0.5f) < 1e-5f);
    TEST_CHECK(lb.pad_x == 0);
    TEST_CHECK(lb.pad_y == 40);

    // Check top gray border and center content pixel
    const cv::Vec3b top_border = dst.at<cv::Vec3b>(0, 160);
    TEST_CHECK(top_border == cv::Vec3b(114, 114, 114));
    const cv::Vec3b center_px = dst.at<cv::Vec3b>(160, 160);
    TEST_CHECK(center_px == cv::Vec3b(10, 20, 30));

    // Round-trip coordinate restoration: (100, 120) in 640x480 -> (50, 60 + 40) = (50, 100) in 320x320
    std::vector<steel::Detection> dets;
    dets.push_back(steel::Detection{50.0f, 100.0f, 200.0f, 220.0f, 0.9f, 0});
    steel::restore_coordinates(dets, lb);

    TEST_CHECK(std::fabs(dets[0].x1 - 100.0f) < 1e-3f);
    TEST_CHECK(std::fabs(dets[0].y1 - 120.0f) < 1e-3f);
    TEST_CHECK(std::fabs(dets[0].x2 - 400.0f) < 1e-3f);
    TEST_CHECK(std::fabs(dets[0].y2 - 360.0f) < 1e-3f);
}

void test_nms_per_class_and_agnostic() {
    std::vector<steel::Detection> cands = {
        {10.0f, 10.0f, 50.0f, 50.0f, 0.90f, 0},  // top crazing
        {12.0f, 12.0f, 52.0f, 52.0f, 0.70f, 0},  // overlaps heavily, same class -> suppress
        {11.0f, 11.0f, 51.0f, 51.0f, 0.85f, 1},  // overlaps heavily, diff class -> keep in per-class
        {100.0f, 100.0f, 140.0f, 140.0f, 0.60f, 2}
    };

    auto per_cls = steel::nms(cands, 0.45f, false);
    TEST_CHECK(per_cls.size() == 3);
    TEST_CHECK(per_cls[0].score == 0.90f && per_cls[0].class_id == 0);
    TEST_CHECK(per_cls[1].score == 0.85f && per_cls[1].class_id == 1);
    TEST_CHECK(per_cls[2].score == 0.60f && per_cls[2].class_id == 2);

    auto agnostic = steel::nms(cands, 0.45f, true);
    TEST_CHECK(agnostic.size() == 2);
    TEST_CHECK(agnostic[0].score == 0.90f);
    TEST_CHECK(agnostic[1].score == 0.60f);
}

void test_yolov5_lite_decoder_math() {
    steel::YoloV5LiteDecoder decoder(320, 320);
    TEST_CHECK(decoder.total_anchors() == 6300);

    steel::RawOutput raw;
    raw.shape = {1, 6300, 11};
    raw.data.assign(6300 * 11, 0.0f);

    // Helper to index into (layer, anchor, gy, gx)
    auto set_pred = [&](int offset_base, int ny, int nx, int a, int gy, int gx,
                        float tx, float ty, float tw, float th, float obj, int cls, float prob) {
        const int idx = offset_base + ((a * ny + gy) * nx + gx);
        float* p = raw.data.data() + static_cast<size_t>(idx) * 11;
        p[0] = tx;
        p[1] = ty;
        p[2] = tw;
        p[3] = th;
        p[4] = obj;
        p[5 + cls] = prob;
    };

    // Layer 0: stride=8, ny=40, nx=40, base=0
    // Anchor 1 = (16, 30), gy=5, gx=10:
    // cx = (0.5*2 - 0.5 + 10) * 8 = 84; cy = (0.5*2 - 0.5 + 5) * 8 = 44
    // w  = (0.5*2)^2 * 16 = 16; h = (0.5*2)^2 * 30 = 30
    // score = 0.8 * 0.30 = 0.24 (> crazing thresh 0.20)
    set_pred(0, 40, 40, 1, 5, 10, 0.5f, 0.5f, 0.5f, 0.5f, 0.8f, 0, 0.30f);

    // Overlapping duplicate box in Layer 0, anchor 0, same cell, lower score (0.7 * 0.30 = 0.21) -> IoU ~0.88, NMS drops it
    set_pred(0, 40, 40, 0, 5, 10, 0.5f, 0.5f, 0.6f, 0.75f, 0.7f, 0, 0.30f);

    // Layer 1: stride=16, ny=20, nx=20, base=4800
    // Class 2 (patches) score = 0.9 * 0.8 = 0.72 <= per-class thresh 0.85 -> must be filtered out
    set_pred(4800, 20, 20, 0, 3, 3, 0.5f, 0.5f, 0.5f, 0.5f, 0.9f, 2, 0.80f);

    // Layer 2: stride=32, ny=10, nx=10, base=4800+1200=6000
    // Anchor 2 = (373, 326), gy=4, gx=6:
    // cx = (0.75*2 - 0.5 + 6) * 32 = 7 * 32 = 224
    // cy = (0.25*2 - 0.5 + 4) * 32 = 4 * 32 = 128
    // w  = (0.25*2)^2 * 373 = 0.25 * 373 = 93.25
    // h  = (0.25*2)^2 * 326 = 0.25 * 326 = 81.5
    // Class 5 (scratches) score = 0.95 * 0.90 = 0.855 > 0.80 -> kept!
    set_pred(6000, 10, 10, 2, 4, 6, 0.75f, 0.25f, 0.25f, 0.25f, 0.95f, 5, 0.90f);

    steel::LetterBoxInfo lb;
    lb.scale = 1.6f;  // 200x200 -> 320x320
    lb.pad_x = 0;
    lb.pad_y = 0;
    lb.orig_w = 200;
    lb.orig_h = 200;
    lb.target_w = 320;
    lb.target_h = 320;

    steel::DetectorConfig cfg;
    cfg.input_w = 320;
    cfg.input_h = 320;
    cfg.iou_threshold = 0.45f;
    cfg.class_conf_thresholds = steel::YoloV5LiteDecoder::default_class_thresholds();

    std::vector<steel::Detection> dets = decoder.decode(raw, lb, cfg);
    TEST_CHECK(dets.size() == 2);

    // Sorted descending by score: scratches (0.855) first, crazing (0.24) second
    TEST_CHECK(dets[0].class_id == 5);
    TEST_CHECK(std::fabs(dets[0].score - 0.855f) < 1e-4f);
    // Expected in 320x320: x1=224-46.625=177.375, y1=128-40.75=87.25, x2=270.625, y2=168.75 -> /1.6
    TEST_CHECK(std::fabs(dets[0].x1 - (177.375f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[0].y1 - (87.25f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[0].x2 - (270.625f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[0].y2 - (168.75f / 1.6f)) < 1e-2f);

    TEST_CHECK(dets[1].class_id == 0);
    TEST_CHECK(std::fabs(dets[1].score - 0.24f) < 1e-4f);
    // Expected in 320x320: x1=84-8=76, y1=44-15=29, x2=92, y2=59 -> /1.6
    TEST_CHECK(std::fabs(dets[1].x1 - (76.0f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[1].y1 - (29.0f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[1].x2 - (92.0f / 1.6f)) < 1e-2f);
    TEST_CHECK(std::fabs(dets[1].y2 - (59.0f / 1.6f)) < 1e-2f);
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };

    const TestCase tests[] = {
        {"test_bounded_queue_fifo_and_block", test_bounded_queue_fifo_and_block},
        {"test_bounded_queue_drop_oldest", test_bounded_queue_drop_oldest},
        {"test_bounded_queue_close_unblocks_waiters", test_bounded_queue_close_unblocks_waiters},
        {"test_letterbox_and_restore_roundtrip", test_letterbox_and_restore_roundtrip},
        {"test_nms_per_class_and_agnostic", test_nms_per_class_and_agnostic},
        {"test_yolov5_lite_decoder_math", test_yolov5_lite_decoder_math},
    };

    int failed = 0;
    for (const auto& t : tests) {
        try {
            t.fn();
            std::cout << "[PASS] " << t.name << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "[FAIL] " << t.name << ": " << e.what() << "\n";
        }
    }

    std::cout << "Summary: " << (6 - failed) << "/6 tests passed.\n";
    return failed == 0 ? 0 : 1;
}
