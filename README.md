# Steel Defect Deploy C++ (`steel_defect_deploy_cpp`)

面向工业钢材表面缺陷检测（NEU-DET 六分类）的 **Linux C++17 端侧多后端异步推理部署框架**。  
工程采用 **策略模式 (`IDetector`) + 三线程异步流水线 (`AsyncPipeline`) + 有界线程安全队列 (`BoundedSafeQueue`) + RAII 资源管理**，同时兼容 **ONNX Runtime (YOLOv8 / YOLOv5-Lite)** 与 **MNN FP16 (YOLOv5-Lite)**，并内置分段耗时统计、CSV 报表导出及 CTest 单元测试套件。

---

## 1. 系统架构与三线程数据流

```text
  [IFrameSource: ImageDir / Video / Camera]
                     │
                     ▼  (Stage 1 Thread: 采图 + LetterBox + HWC->CHW)
         BoundedSafeQueue<PrepTask>   (kBlock / kDropOldest, stop+notify_all)
                     │
                     ▼  (Stage 2 Thread: IDetector::run_raw 专用推理线程)
         BoundedSafeQueue<InferTask>  (std::move 零深拷贝传递 cv::Mat & RawOutput)
                     │
                     ▼  (Stage 3 Thread: 解码 + 手写 NMS + 坐标反算 + 画框/落盘/统计)
       [Annotated Images + CSV Report + Stage Timing Stats]
```

### 核心设计亮点
- **多后端策略抽象 (`include/detector.hpp`)**：`IDetector` 将任意检测器统一拆分为 `preprocess()`、`run_raw()`、`decode_and_postprocess()` 三个正交接口。切换模型（YOLOv8n ↔ YOLOv5-Lite）或推理引擎（ONNX Runtime ↔ MNN）时，前处理、手写 NMS、三线程管线和统计模块**零改动复用**。
- **双模式有界安全队列 (`include/bounded_safe_queue.hpp`)**：支持 `kBlock`（离线批处理零丢帧背压）与 `kDropOldest`（实时摄像头低延迟防 OOM 爆内存），带谓词防虚假唤醒与 `close()` 全员唤醒优雅退出。
- **异常安全与 RAII (`src/async_pipeline.cpp`)**：`ThreadJoinGuard` 保证异常或 `SIGINT` 触发时三线程必然 `join()`；工作线程通过 `std::exception_ptr` 将异常安全回传至主线程。
- **精确对齐 YOLOv5-Lite (`src/yolov5_lite_decoder.cpp`)**：完整复刻 3 层特征图（stride 8/16/32）Anchor 解码、分类别置信度阈值（如 `crazing=0.20`）与手写 `NMS`。

---

## 2. 目录结构

```text
steel_defect_deploy_cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── types.hpp                     # 公共结构体（Detection, LetterBoxInfo, StageTiming 等）
│   ├── bounded_safe_queue.hpp        # 有界线程安全队列（阻塞 / 丢旧帧 / close）
│   ├── preprocess.hpp                # LetterBox 与行指针 HWC->CHW 归一化
│   ├── postprocess.hpp               # IoU、手写 NMS、坐标反变换与画框渲染
│   ├── detector.hpp                  # IDetector 抽象接口
│   ├── yolov5_lite_decoder.hpp       # YOLOv5-Lite 320x320 Anchor 解码器
│   ├── ort_yolov8_detector.hpp       # ONNX Runtime YOLOv8 后端
│   ├── ort_yolov5lite_detector.hpp   # ONNX Runtime YOLOv5-Lite 后端
│   ├── mnn_yolov5lite_detector.hpp   # MNN FP16 YOLOv5-Lite 后端 (-DENABLE_MNN=ON)
│   ├── frame_source.hpp              # 图像目录 / 视频文件 / 摄像头输入源抽象
│   └── async_pipeline.hpp            # 同步基准 / 三线程异步流水线 / CSV 导出
├── src/                              # 对应实现与 CLI 主入口 main.cpp
├── tests/test_core.cpp               # CTest 单元测试（队列并发、LetterBox、NMS、解码数学）
├── models/                           # 模型文件存放处
├── data/                             # 测试图像存放处
└── output/                           # 画框输出与 CSV 报告
```

---

## 3. 编译与测试

### 环境依赖
- C++17 编译器（实测 `g++ 11.4.0`）、CMake `>= 3.16`（实测 `3.22.1`）
- OpenCV（实测 `4.5.4`）
- ONNX Runtime C++ SDK（实测 `1.23.2`，可通过 `-DORT_ROOT=<path>` 指定）
- （可选）MNN C++ SDK（通过 `-DENABLE_MNN=ON -DMNN_ROOT=<path>` 启用）

### Release 编译与运行单元测试
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -V
```

### Debug 编译（供 GDB / Valgrind 体检）
```bash
cmake -S . -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug -j$(nproc)
```

### 启用 MNN 后端编译（接入 `best_se.mnn` / `best_se_sim_fp32.mnn` 后）
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MNN=ON -DMNN_ROOT=/path/to/MNN
cmake --build build -j$(nproc)
```

---

## 4. 常用运行命令

```bash
# 1. 单图推理 + 20 次分段耗时 Benchmark（YOLOv8n ONNX）
./build/steel_detect \
  --model ./models/best.onnx \
  --input ./data/test.jpg \
  --output ./output --warmup 3 --runs 20 --threads 4

# 2. 目录批量：同步基线 vs 三线程异步流水线对比 + 导出 CSV 报告
./build/steel_detect \
  --model ./models/best.onnx \
  --input ./data/neu_det_sample60 \
  --output ./output/sample60 \
  --mode compare --threads 2 --queue-cap 8 --save-images 1 \
  --csv ./output/defect_report.csv

# 3. 运行 YOLOv5-Lite (320x320 + 分类别阈值) ONNX / MNN 后端
./build/steel_detect \
  --backend ort_v5lite --use-v5-class-thresh \
  --model ./models/best_se_sim.onnx \
  --input ./data/neu_det_sample60 --mode compare --threads 2

./build/steel_detect \
  --backend mnn_v5lite --use-v5-class-thresh \
  --model ./models/best_se.mnn \
  --input ./data/neu_det_sample60 --mode async --threads 4 --overflow drop
```

---

## 5. 实测数据记录

### 单元测试验收（`ctest`）
- `steel_core_tests`: **6/6 PASS**（耗时 0.07s）：
  1. `test_bounded_queue_fifo_and_block`
  2. `test_bounded_queue_drop_oldest`
  3. `test_bounded_queue_close_unblocks_waiters`
  4. `test_letterbox_and_restore_roundtrip`
  5. `test_nms_per_class_and_agnostic`
  6. `test_yolov5_lite_decoder_math`

### 60 张 NEU-DET 六类均衡样本实测（Sync vs 3-Thread Async）
- 测试环境：WSL2 Ubuntu 22.04，AMD Ryzen 5 3500U（4 物理核 / 8 逻辑线程）
- 测试模型：YOLOv8n `best.onnx`（`1×3×640×640` → `1×10×8400`，`conf=0.25, iou=0.45`，`queue-cap=8`，`save-images=1`）

| 实验配置 | 模式 | 处理帧数 | 墙钟总耗时 (ms) | 吞吐 (FPS) | 加速比 | 检出一致性 |
|---|---|---|---|---|---|---|
| `--threads 2` | Sync Baseline | 60 | 15395.01 | 3.90 | 1.00x | 108 缺陷 (零丢帧) |
| `--threads 2` | 3-Thread Async | 60 | 14767.73 | 4.06 | **1.04x** | **PASS (精确一致 108)** |
| `--threads 4` | Sync Baseline | 60 | 10628.48 | **5.65** | 1.00x | 108 缺陷 (零丢帧) |
| `--threads 4` | 3-Thread Async | 60 | 10900.95 | 5.50 | 0.98x | **PASS (精确一致 108)** |

### 360 张完整 NEU-DET 验证集实测（YOLOv5-Lite `best_se_sim.onnx`，`ort_v5lite`）
- 测试模型：YOLOv5-Lite-s (+SE) `best_se_sim.onnx`（`1×3×320×320` → `1×6300×11`，`--use-v5-class-thresh`，`iou=0.45`，`queue-cap=8`，`save-images=1`）
- 验证集构成：`new_datasets/valid/images` 共 360 张（`crazing=64, inclusion=64, patches=52, pitted_surface=53, rolled-in_scale=62, scratches=65`）

| 实验配置 | 模式 | 处理帧数 | 墙钟总耗时 (ms) | 吞吐 (FPS) | 加速比 | 检出一致性 |
|---|---|---|---|---|---|---|
| `--threads 2` | Sync Baseline | 360 | 11694.58 | 30.78 | 1.00x | 798 缺陷 (零丢帧) |
| `--threads 2` | 3-Thread Async | 360 | **9208.22** | **39.10** | **1.27x** | **PASS (精确一致 798)** |
| `--threads 4` | Sync Baseline | 360 | 9429.81 | 38.18 | 1.00x | 798 缺陷 (零丢帧) |
| `--threads 4` | 3-Thread Async | 360 | 9291.59 | 38.74 | 1.01x | **PASS (精确一致 798)** |

> 360 张全量六类检出分布（Sync 与 Async 完全一致）：`crazing=231 | inclusion=194 | patches=96 | pitted_surface=62 | rolled-in_scale=154 | scratches=61`（总计 **798** 个缺陷框）。

**工程归因分析（面试高频复盘点）**：
1. **阶段占比决定流水线加速上限（阿姆达尔定律）**：640×640 YOLOv8n 推理占总耗时 ~97%，异步仅能拿到 `1.04x`；而 320×320 YOLOv5-Lite 单帧推理降至 `~25 ms` 后，读图+前处理+画框落盘占比上升，三线程异步在 `--threads 2` 下立刻把端到端吞吐从 `30.78 FPS` 推高到 **`39.10 FPS`（1.27x 提速）**。
2. **核数分配比盲目拉满推理线程更关键**：在 4 物理核 CPU 上，`--threads 2` 异步（2 核给推理、2 核给前后处理 I/O）总耗时 `9208.22 ms`（`39.10 FPS`），**反而优于** `--threads 4` 异步（`9291.59 ms` / `38.74 FPS`）与 `--threads 4` 同步（`9429.81 ms` / `38.18 FPS`），实证了合理划分线程核心避免 OpenCV/ORT 抢核与缓存抖动的工程价值。

## 6. 多后端对比（ORT vs MNN FP32 vs MNN FP16）与内存安全体检


### 1. 360 张 NEU-DET 全量多后端对标（`--threads 2 --use-v5-class-thresh --save-images 0`）

| 推理后端 & 模型 | 执行模式 | 总耗时 (ms) | 吞吐量 (FPS) | 平均纯推理 | 总检出数 | 精度对齐结论 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ORT FP32** (`best_se_sim.onnx`) | 3-Thread Async | 7,974.19 | **45.15** | 22.03 ms | **798** | 基准 (100%) |
| **MNN FP32** (`best_se_sim_fp32.mnn`) | Sync / Async | 10,845.51 / 10,938.40 | 33.19 / 32.91 | 28.53 / 30.25 ms | **798** | **100% 绝对一致** |
| **MNN FP16** (`best_se.mnn`) | Sync / Async | 13,386.31 / 12,450.48 | 26.89 / 28.91 | 34.47 / 34.44 ms | 786 | 98.50% (-12 框) |

> **工程归因**：
> 1. **MNN FP32 与 ORT FP32 798/798 完全一致**，验证了多后端接口解耦与 `MNN::Tensor::CAFFE` 数据排布零误差。
> 2. **x86_64 下 FP16 反而慢于 FP32**：MNN 运行日志显示 `fp16:0`（x86 CPU 无原生 FP16 硬件指令集），开启 `Precision_Low` 触发额外反量化转换开销；FP16 带宽减半收益需在 ARM/NPU 端侧体现。
> 3. **FP16 掉框定位**：缺失的 12 框集中在 `crazing`(-6)、`patches`(-4，阈值 0.85)、`scratches`(-2，阈值 0.80)，因 FP16 10-bit 尾数在高置信度阈值边缘产生轻微舍入导致跌破阈值。

### 2. Valgrind 内存零泄漏体检（Debug `-g` 构建）
- `steel_tests`（6/6 单元测试）：`definitely lost: 0 bytes`, `indirectly lost: 0 bytes`, `ERROR SUMMARY: 0 errors`
- `ort_v5lite`（3 线程异步流水线）：`definitely lost: 0 bytes`, `indirectly lost: 0 bytes`
- `mnn`（3 线程异步流水线）：`definitely lost: 0 bytes`, `indirectly lost: 0 bytes`
