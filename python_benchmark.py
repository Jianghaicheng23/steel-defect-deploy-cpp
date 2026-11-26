#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ONNX Runtime Baseline Benchmark for YOLOv5-Lite
Evaluates exactly 360 validation images with the exact NEU-DET anchor decoding & NMS.
"""

import os
import sys
import time
import glob
import numpy as np
import cv2
import onnxruntime as ort

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_SIZE = 320
CLASS_CONF_THRESH = [0.20, 0.60, 0.85, 0.45, 0.35, 0.80]
IOU_THRES = 0.45

STRIDES = [8, 16, 32]
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]]
]

def preprocess(image):
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 114
    x_offset = (IMG_SIZE - new_w) // 2
    y_offset = (IMG_SIZE - new_h) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_img
    
    # HWC -> CHW, 1/255.0, BGR preserved
    img_tensor = padded_img.astype(np.float32) * (1.0 / 255.0)
    img_tensor = np.ascontiguousarray(np.transpose(img_tensor, (2, 0, 1))[None, ...])
    return img_tensor, scale, (new_w, new_h), (x_offset, y_offset)

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def postprocess(raw_out, scale, offset, orig_shape):
    h, w = orig_shape[:2]
    xo, yo = offset
    data = raw_out # (6300, 11)
    cand_by_cls = [[] for _ in range(6)]
    row_idx = 0

    for l in range(3):
        stride = STRIDES[l]
        ny = nx = IMG_SIZE // stride
        for a in range(3):
            anc_w, anc_h = ANCHORS[l][a]
            for gy in range(ny):
                for gx in range(nx):
                    p = data[row_idx]
                    row_idx += 1
                    obj = p[4]
                    cls_probs = p[5:]
                    best_cls = int(np.argmax(cls_probs))
                    score = float(obj * cls_probs[best_cls])
                    if score > CLASS_CONF_THRESH[best_cls]:
                        cx = (p[0] * 2.0 - 0.5 + gx) * stride
                        cy = (p[1] * 2.0 - 0.5 + gy) * stride
                        tw = p[2] * 2.0
                        th = p[3] * 2.0
                        bw = (tw * tw) * anc_w
                        bh = (th * th) * anc_h
                        x1 = max(0.0, min(float(w - 1), (cx - bw * 0.5 - xo) / scale))
                        y1 = max(0.0, min(float(h - 1), (cy - bh * 0.5 - yo) / scale))
                        x2 = max(0.0, min(float(w - 1), (cx + bw * 0.5 - xo) / scale))
                        y2 = max(0.0, min(float(h - 1), (cy + bh * 0.5 - yo) / scale))
                        cand_by_cls[best_cls].append((score, x1, y1, x2, y2))

    all_dets = []
    for c in range(6):
        boxes = sorted(cand_by_cls[c], key=lambda x: x[0], reverse=True)
        keep = []
        for b in boxes:
            suppress = False
            for k in keep:
                if iou(b[1:], k[1:]) > IOU_THRES:
                    suppress = True
                    break
            if not suppress:
                keep.append(b)
                all_dets.append((c, b[0], b[1], b[2], b[3], b[4]))
    return all_dets

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Python YOLOv5-Lite Benchmark")
    parser.add_argument("--model", type=str, default="models/best_se_sim.onnx", help="Model path")
    parser.add_argument("--images", type=str, default="../YOLOv5-Lite-master/new_datasets/valid/images", help="Images folder")
    parser.add_argument("--threads", type=int, default=2, help="Intra-op threads")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--save-csv", type=str, default="output/python_benchmark_results.csv", help="CSV output")
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"Loading ONNX model: {args.model} with {args.threads} intra-op threads...")
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img_files = sorted(glob.glob(os.path.join(args.images, "*.*")))
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(img_files)} images in {args.images}")

    if len(img_files) == 0:
        print("No images found! Exiting.")
        return

    # Warmup
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})
    print(f"Warmup {args.warmup} iterations complete.")

    total_read_ms = 0.0
    total_pre_ms = 0.0
    total_infer_ms = 0.0
    total_post_ms = 0.0
    class_counts = {c: 0 for c in CLASSES}
    total_detections = 0

    t_start = time.perf_counter()

    for idx, path in enumerate(img_files):
        tr0 = time.perf_counter()
        img_bgr = cv2.imread(path)
        tr1 = time.perf_counter()
        if img_bgr is None:
            continue

        # Preprocess
        t0 = time.perf_counter()
        inp, scale, shape, offset = preprocess(img_bgr)
        t1 = time.perf_counter()

        # Inference
        outputs = session.run(None, {input_name: inp})
        raw_out = outputs[0][0]
        t2 = time.perf_counter()

        # Postprocess
        dets = postprocess(raw_out, scale, offset, img_bgr.shape)
        t3 = time.perf_counter()

        total_read_ms += (tr1 - tr0) * 1000.0
        total_pre_ms += (t1 - t0) * 1000.0
        total_infer_ms += (t2 - t1) * 1000.0
        total_post_ms += (t3 - t2) * 1000.0

        for d in dets:
            cls_id = d[0]
            if 0 <= cls_id < len(CLASSES):
                class_counts[CLASSES[cls_id]] += 1
                total_detections += 1

    t_end = time.perf_counter()
    total_wall_ms = (t_end - t_start) * 1000.0
    n = len(img_files)
    fps = (n / (total_wall_ms / 1000.0))

    print("\n================= Python Benchmark Results =================")
    print(f"Images Processed     : {n}")
    print(f"Total Wall Clock Time: {total_wall_ms:.2f} ms")
    print(f"Overall Throughput   : {fps:.2f} FPS")
    print(f"Average Read         : {total_read_ms / n:.2f} ms")
    print(f"Average Preprocess   : {total_pre_ms / n:.2f} ms")
    print(f"Average Inference    : {total_infer_ms / n:.2f} ms")
    print(f"Average Postprocess  : {total_post_ms / n:.2f} ms")
    print(f"Average E2E Latency  : {(total_read_ms + total_pre_ms + total_infer_ms + total_post_ms) / n:.2f} ms")
    print("---------------- Class Detections ----------------")
    for c, cnt in class_counts.items():
        print(f"  {c:18s}: {cnt:4d}")
    print(f"  {'Total Detections':18s}: {total_detections:4d}")
    print("============================================================\n")

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"images,{n}\n")
        f.write(f"total_time_ms,{total_wall_ms:.2f}\n")
        f.write(f"fps,{fps:.2f}\n")
        f.write(f"avg_read_ms,{total_read_ms / n:.2f}\n")
        f.write(f"avg_preprocess_ms,{total_pre_ms / n:.2f}\n")
        f.write(f"avg_inference_ms,{total_infer_ms / n:.2f}\n")
        f.write(f"avg_postprocess_ms,{total_post_ms / n:.2f}\n")
        f.write(f"avg_e2e_ms,{(total_read_ms + total_pre_ms + total_infer_ms + total_post_ms) / n:.2f}\n")
        for c, cnt in class_counts.items():
            f.write(f"count_{c},{cnt}\n")
        f.write(f"total_detections,{total_detections}\n")
    print(f"Saved metrics to: {args.save_csv}")

if __name__ == "__main__":
    main()
