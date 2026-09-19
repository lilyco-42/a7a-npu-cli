# a7a_npu — A733 (Radxa Cubie A7A) NPU 统一 CLI 工具箱

把 NPU 的 6 类能力收敛成**一个命令**, 复用 `lyco_agent/tools/` 里已验证的组件。
来源: `lilyco-42/lyco_agent` 的 NPU 工具链 (kws_run.py / npulib / lyco-vipcore.service)。

> 状态: 本目录是 **local source**。NPU 推理需在板端运行 (内核 6.6.98+, VIPLite 2.0.3.2)。
> 当前板子 SD 卡有写入缺陷, **部署/测试请等接上 USB SSD 后**再上板。

## 能力 → 子命令 映射

| 能力 | 子命令 | 底层机制 | 板端资产 | 来源仓库 |
|---|---|---|---|---|
| LLM 推理 | `a7a_npu llm` | llama.cpp + TIM-VX 把 `MUL_MAT` 卸载到 `/dev/galcore` | llama.cpp A733 后端 + GGUF | `reef1994/a733-llama-npu-stack` |
| 视觉-语言 | `a7a_npu vlm` | vpm_run + MobileCLIP NBG | `official_a733.nb` (ShuffleNetV2_uint8) | `petayyyy/a733_npu_driver` |
| 目标检测 | `a7a_npu detect` | vpm_run + YOLOv5/8 NBG (或原厂 demo) | `yolov5s_rt_uint8_a733.nb` | 我们板端实测 `detection num:3` ✓ |
| 图像分类 | `a7a_npu classify` | vpm_run + ResNet/ShuffleNet/MobileNet NBG | Model Zoo | Radxa `awnpu_model_zoo` |
| 语音(离线) | `a7a_npu kws` | 音频→fbank→NPU enc/dec/joiner→打分 | encoder/decoder/joiner float NBG | `Ronin-1124/cubie-a7a-voice-assistant` |
| 边缘视觉 | `a7a_npu edge` | vpm_run + 零售/pose/OCR NBG | 对应 NBG | `senijus/smart_retail` |

## 快速开始 (板端)

```bash
# 把本目录传到板子 (用 rsftp.py 或 scp), 进 workdir
cd /home/radxa/kws-min          # 含 vpm_run + lib + models

# 目标检测 (原厂 demo, 直接出框)
python3 a7a_npu.py detect --demo model/dog.jpg

# 或通用 NBG 路径
python3 a7a_npu.py detect --model models/yolov5s_rt_uint8_a733.nb model/dog.jpg

# 图像分类
python3 a7a_npu.py classify models/official_a733.nb model/cat.jpg

# KWS 链路自检 (不给 wav 用 1s 扫频音)
python3 a7a_npu.py kws

# LLM (需先部署 reef1994/a733-llama-npu-stack)
python3 a7a_npu.py llm --model /path/qwen2.5-0.5b.gguf --prompt "你好"

# 只看将要执行的命令, 不真正跑
python3 a7a_npu.py detect --dry-run model/dog.jpg
```

## 设计说明

- **`run_nbg(workdir, model, inputs)`** — 通用 NBG 运行器: 写 `sample.txt` →
  `vpm_run -s sample.txt -l N`。这是我们在板端验证过的路径
  (YOLOv5s uint8 实测 `detection num:3`, 官方量化 NBG `ret=0` + IRQ)。
- **`run_kws()`** — 移植自 `lyco_agent/tools/kws_run.py` 的完整 KWS 链路
  (音频→fbank(80×29)→NPU encoder→decoder→joiner→263 vocab logits top5)。
- **`lyco-vipcore.service`** — 负责把 `3600000.npu` 从 galcore 切到 vipcore
  (**绝不能 rmmod galcore**, 会 panic); 启动 NPU 前需先 `systemctl start lyco-vipcore`。
- 根因/历史见 `radxa_utlra/docs/a733-npu-three-layer-rootcause.md`
  (量化通道在 Radxa 原厂 `6.6.98-4-aw2511` 未接好, 换 Rabs9 `6.6.98+` 后全通)。

## 依赖

- 板端: VIPLite 2.0.3.2 + `/dev/vipcore`, 已编译的 `vpm_run`, `libVIPhal.so`/`libNBGlinker.so`, 对应 `.nb` 模型
- Python: `numpy` (仅 KWS 需要)
- `kws` 还需 `kws-repo` 的 token 表 (可选, 影响 top5 可读性)
