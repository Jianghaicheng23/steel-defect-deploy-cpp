# 排障与工程归因案例（Troubleshooting）

## 案例 1：MNN FP16 在 x86_64 下反慢于 FP32 且检出少 12 框
- **异常现象**：360 张 NEU-DET 验证集上，`best_se.mnn` (FP16) 检出 786 框（比 ORT FP32 少 12 框），且平均推理耗时 34.47 ms 反而慢于 FP32 模型（28.53 ms）。
- **排查手段**：引入 `best_se_sim_fp32.mnn` 做控变量 A/B 对比，逐类统计 6 类缺陷召回差异，并查看 MNN 启动硬件特性日志。
- **根因定位**：
  1. MNN FP32 检出 **798/798（100% 一致）**，排除前处理、`MNN::Tensor::CAFFE` 内存排布与解码代码 Bug；
  2. MNN 日志打印 `fp16:0`，WSL2 x86_64 CPU 无原生 FP16 算力单元，开启 `Precision_Low` 引入权重反量化额外开销；
  3. 缺失 12 框集中在 `crazing`(-6)、`patches`(-4，阈值 0.85)、`scratches`(-2，阈值 0.80)，因 FP16 10-bit 尾数在高置信度阈值边缘向下舍入。
- **结论与取舍**：x86 宿主机以 FP32 做零漂移基准；ARM/NPU 端侧保留 FP16 换取 50% 体积与内存带宽收益，并对边缘高阈值类别微调校准。

## 案例 2：Python 重构后处理 300x 提速与多核异步核数抢占调优
- **异常现象**：原 Python 端到端仅 21.94 FPS；重构三线程异步后，4 物理核 CPU 上设置推理 `--threads 4` 时异步吞吐（38.74 FPS）反不如 `--threads 2`（39.10 FPS），且 640×640 YOLOv8n 异步加速比仅 1.04x。
- **排查手段**：使用 `std::chrono::steady_clock` 拆解 `read / prep / infer / post / draw+save` 五阶段耗时并对比线程配比。
- **根因定位**：
  1. 原 Python 瓶颈在于 6300 个 Anchor 遍历与 PyObject 动态分配，后处理高达 22.05 ms；
  2. 受阿姆达尔定律约束，YOLOv8n 推理占比 97% 限制并行收益，轻量 YOLOv5-Lite 推理降至 ~25 ms 后异步稳定达 1.27x（落盘模式）/ 45.15 FPS（纯检测）；
  3. 4 核 CPU 上给推理开 4 线程会与 Stage 1/3 前后处理线程抢核，引发上下文切换与 Cache 抖动。
- **修复与验证**：C++ 行指针前处理 + 前置置信度过滤 + `std::vector<uint8_t>` 手写 NMS 将后处理压至 **0.05 ms**；推理线程收敛为 `--threads 2`，吞吐反超 `--threads 4` 且 Sync/Async 798 框精确一致。

## 案例 3：多后端资源生命周期、Valgrind 误报判读与 GDB 多线程死锁定位
- **风险与现象**：三线程流水线同时管理 ORT/MNN 底层句柄与跨线程有界队列；Debug `-g` 模式用 Valgrind 体检 MNN 时出现 `Conditional jump depends on uninitialised value` 告警。
- **排查工具**：`valgrind --leak-check=full --track-origins=yes` 与 `gdb`（`info threads` / `thread apply all bt` / `p mtx` 查看 `__owner` LWP）。
- **根因定位与加固**：
  1. 借助 `-g` 符号栈与 `--track-origins=yes` 定位告警源头在 `libMNN.so` 内部 `MNN::GeometryImageOp::onCompute`（非业务代码），单测 / ORT / MNN 三路均验证 **`definitely lost: 0 bytes, indirectly lost: 0 bytes`**；
  2. 严格按 `host_tensor → releaseSession → Interpreter::destroy` 逆序析构防 Use-After-Free，并在基类 `IDetector` 中 `= delete` 禁用拷贝/移动防 Double Free；
  3. 队列采用条件变量 Lambda 谓词 + `close()` 全员唤醒 + RAII `ThreadJoinGuard` 杜绝退出死锁，并用 GDB 验证了通过 `thread apply all bt` 与互斥锁 `__owner` LWP 锁定 AB-BA 环路等待行号的排障链路。
