# a7a_npu — A733 / Radxa Cubie A7A 统一 CLI

这是一个**薄适配层**：把 Radxa 官方 A7A NPU 文档里的工具链收敛成 CLI；不重写 ACUITY、VIPLite、NBinfo、G2D 或官方 Model Zoo 的后处理器。

官方文档目录：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/>

> 已完成的是**工具级 CLI 覆盖**，不是“23 个 Model Zoo 项目全部自动下载、预处理、后处理并逐个实测”。
> NPU 真实推理仍在 A7A 板端运行；当前板子的 NPU 量化已在 Rabs9 `6.6.98+` 内核下实测通过，但 SD 卡有写入缺陷，部署/测试前应先迁移到 USB SSD。

## 覆盖矩阵

| 官方能力 | CLI | 状态 |
|---|---|---|
| A7A 硬件资源只读盘点（NPU/G2D/VE2/GPU/DMA heap/温度/内存/CPU 频率） | `a7a_npu hardware --json` | ✅ 只读 CLI |
| VIPLite `vpm_run` 编译、NBG 运行、循环/超时/设备/core/bypass/top5/保存输出/层 profiling/preload/分段/layer dump | `a7a_npu vpm ...` | ✅ 参数已覆盖；板端 YOLOv5 和量化 NBG 已实测 |
| NBinfo：brief/header/layers/operations/input/output/memory/detail/flash | `a7a_npu nbinfo ...` | ✅ 薄封装；需要 PC 端官方 `nbinfo` |
| ACUITY Docker 环境：检查镜像、载入 tar、进入容器 | `a7a_npu acuity env ...` | ✅ 薄封装；必须在 x86 Linux + Docker |
| ACUITY：导入、量化、推理、OpenVX/NBG 导出 | `a7a_npu acuity import/quantize/infer/export ...` | ✅ 薄封装；必须有官方 A733 v2.0.10.2 镜像 |
| ACUITY：KLD、MLE、entropy、hybrid 混合量化 | `a7a_npu acuity quantize --kld --mle --compute-entropy --hybrid` | ✅ 参数已覆盖 |
| Vivante IDE/OpenVX/NBG PC 模拟 | `a7a_npu acuity simulate ...` | ✅ 薄封装；需要 Vivante IDE |
| Model Zoo 目录、官方包下载、选择性解压 | `a7a_npu model-zoo ...` | ✅ 23 项目录；各项目仍需自己的输入/后处理 |
| 离线语音助手 KWS/ASR/TTS | `a7a_npu voice ...` / `kws` | ✅ 委托官方 `assistant_fast.py`；需要语音助手完整资产 |
| G2D 旋转/镜像、格式转换、缩放/下采样、纯色填充 | `a7a_npu g2d ...` | ✅ 薄封装官方 C 示例；需编译示例 |
| TIM-VX / llama.cpp LLM | `a7a_npu llm ...` | ⚠️ 仅 runner wrapper；不是“LLM 整体都跑 NPU”的保证 |
| YOLOv5 检测 | `a7a_npu detect --demo ...` | ✅ 我们板端实测 3 个检测框 |
| 任意 Model Zoo 的图片预处理/后处理 | 各项目官方 demo | ⚠️ 不假装统一；raw `vpm_run` 要 `.dat/.tensor` |

官方 Model Zoo 共 23 项：YOLO11/Seg/Pose、YOLOv8/Seg/Pose、YOLOv3、YOLOv5、YOLO26、YOLOX、RetinaFace、PPSeg、MobileNetV1/V2、ResNet50、ResNet50V2、DenseNet121、SqueezeNet、LeNet、CLIP、Zipformer、Lite Transformer、LSTM。

## 常用命令

### 1. 板端健康检查

```bash
# 全硬件资源盘点：NPU/G2D/VE2/GPU/DMA heap/温度/内存/CPU 频率
python3 a7a_npu.py hardware --json

# NPU/G2D/IRQ 专项检查
python3 a7a_npu.py info --json
python3 a7a_npu.py info --strict
```

`hardware` 和 `info` 都是只读命令，不会加载模块、不改 sysfs、不写磁盘。`info` 的 debugfs 时钟/电源详情需要 root；设备节点存在只代表驱动枚举成功，真正可用性仍要用 `vpm run` 或官方 demo 验证。

### 2. vpm_run

```bash
# 官方 sample.txt：支持多网络、多个 input、golden、output
python3 a7a_npu.py vpm run --sample sample.txt --loop 10 --show-top5 1

# 生成最小 sample 并运行
python3 a7a_npu.py vpm run --model models/joiner_float_a733.nb \
  models/in0.dat models/in1.dat --loop 1

# 官方调试/性能参数
python3 a7a_npu.py vpm run --sample sample.txt --layer-profile-dump \
  --preload --op-segment 10,20 --layer-dump -1

# 编译官方 ai-sdk 示例
python3 a7a_npu.py vpm build --source /path/ai-sdk/examples/vpm_run --install
```

### 3. NBinfo

```bash
nbinfo_path=/path/to/nbinfo
python3 a7a_npu.py nbinfo --binary "$nbinfo_path" --all network_binary.nb
python3 a7a_npu.py nbinfo --binary "$nbinfo_path" --header network_binary.nb
python3 a7a_npu.py nbinfo --binary "$nbinfo_path" --input network_binary.nb
python3 a7a_npu.py nbinfo --binary "$nbinfo_path" --output network_binary.nb
python3 a7a_npu.py nbinfo --binary "$nbinfo_path" --memory --detail-memory network_binary.nb
```

NBinfo 能检查 target、版本、层/算子数量、输入输出 shape/量化格式、内存池和详细内存画像。

### 4. ACUITY（x86 Docker）

```bash
# 检查/载入/进入 A733 官方镜像
python3 a7a_npu.py acuity env check
python3 a7a_npu.py acuity env load --tar ubuntu-npu_v2.0.10.2.tar
python3 a7a_npu.py acuity env shell --workspace "$PWD"

# 原模型导入
python3 a7a_npu.py acuity import models/MobileNetV2_Imagenet \
  --script-dir /workspace/ai-sdk/models

# 量化、KLD、MLE、熵统计、混合量化
python3 a7a_npu.py acuity quantize models/MobileNetV2_Imagenet \
  --qtype int16 --iterations 10 --kld --mle --compute-entropy --hybrid

# PC 侧精度验证
python3 a7a_npu.py acuity infer models/MobileNetV2_Imagenet --qtype int16

# 导出 OpenVX 项目 + NBG
python3 a7a_npu.py acuity export models/MobileNetV2_Imagenet --qtype int16

# Vivante IDE 模拟（OpenVX/NBG 项目）
python3 a7a_npu.py acuity simulate --project models/MobileNetV2_Imagenet/wksp/MobileNetV2_Imagenet_int16_nbg_unify \
  --binary mobilenetv2imagenetint16 --data network_binary.nb \
  --input ../../space_shuttle_224x224.jpg
```

A733 必须使用官方 v2.0 工具链；文档给出的 Docker 镜像是 `ubuntu-npu:v2.0.10.2`，不要拿 T527 v1.8.13 混用。

### 5. Model Zoo

```bash
python3 a7a_npu.py model-zoo list
python3 a7a_npu.py model-zoo list --json
python3 a7a_npu.py model-zoo download --output allwinner-model-zoo.tar.gz
python3 a7a_npu.py model-zoo extract allwinner-model-zoo.tar.gz \
  --output ./zoo --pattern yolov5s_rt_uint8_a733.nb
```

### 6. 官方离线语音助手

```bash
python3 a7a_npu.py voice assistant --from-wav samples/pipe_nihao_xiaorui_openlight_16k.wav
python3 a7a_npu.py voice assistant
python3 a7a_npu.py voice matcha --text "识别完成" --play
# kws 是 voice assistant 的兼容快捷入口
python3 a7a_npu.py kws --from-wav samples/pipe_nihao_xiaorui_openlight_16k.wav
```

官方语音链路是：KWS NPU → ASR NPU → Matcha CPU + HiFi-GAN NPU；NPU 同时只能加载一个网络，不能把多个 NBG 当作并行多模型服务。

### 7. G2D

```bash
python3 a7a_npu.py g2d doctor --json
python3 a7a_npu.py g2d build rotate --source /path/allwinner-g2d-usage-guide
python3 a7a_npu.py g2d run rotate --examples ./bin -- --width 1920 --height 1080
```

G2D 是硬件 2D 加速器，不是 NPU；操作依赖 `/dev/g2d` 和 `/dev/dma_heap/system`。官方示例覆盖旋转/镜像、ARGB→YUV420、缩放/下采样、颜色填充。

### 8. LLM/VLM/分类/边缘视觉的边界

```bash
# YOLOv5：官方 demo 自带图片预处理和后处理，我们已验证
python3 a7a_npu.py detect --demo model/dog.jpg

# 原始 NBG：输入必须是 ACUITY/Vivante 生成的 .dat/.tensor，不是 jpg
python3 a7a_npu.py classify --model models/network_binary.nb input.tensor

# LLM：需先部署外部 llama.cpp/TIM-VX runner
python3 a7a_npu.py llm --model /path/model.gguf --prompt "你好"
```

`vpm_run` 只负责 NBG 网络运行和默认 TOP5，不等于完整的图像分类/检测应用；每个 Model Zoo 项目的预处理和后处理仍以官方 demo 为准。LLM 是 TIM-VX/llama.cpp 的独立路径，A733 NPU 的本职优势是 CNN/视觉和语音，不应把“能卸载部分矩阵乘”写成“完整 Transformer 全在 NPU”。

## 本地验证

```bash
python3 -m py_compile a7a_npu.py
python3 a7a_npu.py capabilities
python3 a7a_npu.py vpm run --dry-run --model models/x.nb models/in.dat
python3 a7a_npu.py nbinfo --dry-run --header models/x.nb
python3 a7a_npu.py acuity quantize --dry-run models/MobileNetV2 --qtype int16 --kld --mle
python3 a7a_npu.py g2d run --dry-run rotate --examples /opt/g2d -- --width 1920 --height 1080
python3 a7a_npu.py voice assistant --dry-run --repo /opt/voice
```

## 依赖和未自动化部分

- A7A 板端：Rabs9 `6.6.98+`、VIPLite 2.0.3.2、`/dev/vipcore`、`vpm_run`、`libVIPhal.so`/`libNBGlinker.so`、对应 A733 `.nb`。
- x86 PC：Docker、官方 A733 ACUITY 镜像、可选 Vivante IDE、`nbinfo`。
- 语音：官方 `cubie-a7a-voice-assistant`、Zipformer demo、NPU 模型、ALSA 耳机麦克风；`--from-wav` 是首选验收路径。
- G2D：官方 `allwinner-g2d-usage-guide` 源码、gcc、`/dev/g2d`、DMA heap。
- CLI 不会自动下载受限的全志 ACUITY 网盘资源，不会猜测模型的输入布局/mean/scale，也不会为 23 个 Model Zoo 项目伪造统一后处理。

## 官方来源

- NPU 开发总览：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/>
- SDK：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-acuity-sdk>
- ACUITY 环境：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-acuity-env>
- ACUITY 使用：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-acuity-usage>
- 量化精度：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-quant-acc-improve>
- vpm_run：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-vpm-run>
- NBinfo：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/cubie-nbinfo>
- G2D：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/g2d-usage-guide>
- Model Zoo：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo>
- 离线语音助手：<https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/voice-assistant>
