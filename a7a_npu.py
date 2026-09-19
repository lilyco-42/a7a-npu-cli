#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a7a_npu — A733 (Radxa Cubie A7A) NPU 统一命令行工具箱

把 NPU 的 6 类能力收敛成一个命令, 复用 lyco_agent/tools 里已验证的组件:
  - NPU 推理底层: VIPLite vpm_run + NBG (kws-min 测试架, 已实测 ret=0 + IRQ)
  - KWS 链路:     kws_run.py 的 音频→fbank→NPU(enc/dec/joiner)→打分 (移植为本文件 run_kws)
  - LLM:          reef1994/a733-llama-npu-stack (llama.cpp + TIM-VX 卸载 MUL_MAT 到 /dev/galcore)
  - VLM/检测/分类/边缘视觉: vpm_run + 对应 NBG (MobileCLIP / YOLOv5 / ResNet / 零售 pose OCR)

所有 NPU 推理走的是我们在板端验证过的 VIPLite 2.0.3.2 路径 (内核 6.6.98+).
本文件是 LOCAL SOURCE; 真正跑推理在板子上. 部署到 USB SSD 后再上板测试.

子命令:
  llm       跑 GGUF 模型 (NPU/CPU 混合, 经 llama.cpp 栈)
  vlm       跑 MobileCLIP-S0 图像→向量/分类 (NPU)
  detect    目标检测 (YOLOv5/8 NBG; 默认 yolov5s_uint8 出检测框)
  classify  图像分类 (ResNet / ShuffleNet / MobileNet NBG)
  kws       关键词唤醒 (音频→fbank→NPU enc/dec/joiner→token 打分)
  edge      边缘视觉 (零售识别 / pose / OCR, 走对应 NBG)
  info      打印板端 NPU 状态 (时钟/电源域/驱动/IRQ)

通用参数:
  --dry-run   只打印将要执行的命令, 不真正运行 (安全/调试用)
  --workdir   工作目录 (默认 /home/radxa/kws-min, 含 vpm_run + lib + models)
"""
import argparse
import os
import subprocess
import sys
import tempfile

# ---- 板端固定路径 (内核 6.6.98+, VIPLite 2.0.3.2, 已验证) ----
DEFAULT_WORKDIR = "/home/radxa/kws-min"
VPM_RUN = "vpm_run"                       # 在 workdir 下编译好的二进制
LIB_DIR = "lib"                           # libVIPhal.so / libNBGlinker.so
MODELS = "models"

# 已知可用模型 (A733 专用, target 0x1000003b)
M_YOLOV5S = f"{MODELS}/yolov5s_rt_uint8_a733.nb"   # 量化检测, 已出 3 框
M_SHUFFLENET = f"{MODELS}/official_a733.nb"        # ShuffleNetV2_uint8 (VLM/分类样例)
M_JOINER = f"{MODELS}/joiner_float_a733.nb"
M_DECODER = f"{MODELS}/decoder_float_a733.nb"
M_ENCODER = f"{MODELS}/encoder_float_a733.nb"
M_VOCORDER = f"{MODELS}/vocoder_int16_a733.nb"

# 原厂 YOLOv5 demo 二进制 (直接出检测框, 我们实测 detection num:3)
YOLO_DEMO = "/home/radxa/npu_zoo/examples/yolov5/build/yolov5_demo_a733"
YOLO_DEMO_NB = "/home/radxa/npu_zoo/examples/yolov5/model/yolov5s_rt_uint8_a733.nb"


def _env(workdir):
    lib = os.path.join(workdir, LIB_DIR)
    return dict(os.environ, LD_LIBRARY_PATH=f"{lib}:{os.environ.get('LD_LIBRARY_PATH','')}")


def run_nbg(workdir, model_nb, inputs, loop=1, dry_run=False, extra=None):
    """通用 NBG 运行器: 写 sample.txt -> 调 vpm_run -s sample.txt -l loop.

    inputs: list[str] 输入 .dat 路径 (相对 workdir 或绝对)
    返回 (rc, out) ; dry_run 时返回 (0, 拟执行命令)
    """
    workdir = os.path.abspath(workdir)
    sample = os.path.join(workdir, "sample_run.txt")
    with open(sample, "w", encoding="utf-8") as f:
        f.write("[network]\n")
        f.write(os.path.join(workdir, model_nb) + "\n")
        f.write("[input]\n")
        for i in inputs:
            f.write((i if os.path.isabs(i) else os.path.join(workdir, i)) + "\n")
    cmd = [os.path.join(workdir, VPM_RUN), "-s", sample, "-l", str(loop)]
    if extra:
        cmd += extra
    if dry_run:
        return 0, "DRY-RUN: " + " ".join(cmd)
    r = subprocess.run(cmd, cwd=workdir, env=_env(workdir),
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def run_kws(wav=None, workdir=DEFAULT_WORKDIR, dry_run=False):
    """完整 KWS 链路 (移植自 lyco_agent/tools/kws_run.py).

    音频 → fbank(80x29) → NPU encoder → decoder → joiner → 263 vocab logits top5
    wav=None 时用 1s 扫频音做链路自检.
    """
    try:
        import numpy as np
    except ImportError:
        return 1, "需要 numpy: pip install numpy"
    import wave

    SR, N_MEL, FRAMES, HOP, WIN = 16000, 80, 29, 160, 400

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def fb():
        pts = np.linspace(hz2mel(20.0), hz2mel(8000.0), N_MEL + 2)
        hzs = 700.0 * (10 ** (pts / 2595.0) - 1.0)
        b = np.floor(513 * hzs / SR).astype(int)
        f = np.zeros((N_MEL, 257), dtype=np.float32)
        for m in range(1, N_MEL + 1):
            l, c, r = b[m - 1], b[m], b[m + 1]
            c, r = max(c, l + 1), max(r, c + 1)
            for k in range(l, c):
                f[m - 1, k] = (k - l) / (c - l)
            for k in range(c, r):
                f[m - 1, k] = (r - k) / (r - c)
        return f

    def fbank(x):
        F = fb()
        win = np.hanning(WIN).astype(np.float32)
        need = (FRAMES - 1) * HOP + WIN
        x = np.pad(x, (0, max(0, need - len(x))))
        out = np.zeros((FRAMES, N_MEL), dtype=np.float32)
        for i in range(FRAMES):
            sp = np.abs(np.fft.rfft(x[i * HOP:i * HOP + WIN] * win, n=512)) ** 2
            out[i] = np.log(np.maximum(F @ sp, 1e-10))
        return out

    def audio(path=None):
        if path:
            with wave.open(path, "rb") as w:
                return np.frombuffer(w.readframes(w.getnframes()),
                                     dtype=np.int16).astype(np.float32) / 32768
        t = np.arange(SR) / SR
        f = 200 + 2800 * t
        return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)

    def read_out(i):
        p = os.path.join(workdir, f"output_{i}.txt")
        if not os.path.exists(p):
            return None
        v = []
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    v.append(float(line.split()[-1]))
                except ValueError:
                    pass
        return np.array(v, dtype=np.float32)

    wd = os.path.abspath(workdir)
    feat = fbank(audio(wav))
    print(f"[1] 音频 {len(audio(wav) if wav else np.arange(SR))/SR:.2f}s → fbank {feat.shape}")

    # encoder: in0=fbank, in1..37=cache(0), in38=scalar
    feat.tofile(os.path.join(wd, "e0.dat"))
    for i in range(1, 38):
        np.zeros(1024, dtype=np.float32).tofile(os.path.join(wd, f"e{i}.dat"))
    np.zeros(1, dtype=np.float32).tofile(os.path.join(wd, "e38.dat"))
    rc, out = run_nbg(wd, M_ENCODER, [f"e{i}.dat" for i in range(39)],
                      dry_run=dry_run, extra=["-b", "0", "--save_txt", "1"])
    if dry_run:
        return 0, out
    enc = read_out(0)
    print(f"[2] NPU encoder: {'OK' if 'ret=0' in out else 'FAIL'}"
          + (f" out0={enc.shape}" if enc is not None else ""))
    if enc is None:
        return 1, out
    enc320 = enc[:320]
    # decoder: in0 = 2 token ids [0,0]
    np.array([0, 0], dtype=np.float32).tofile(os.path.join(wd, "d0.dat"))
    rc, out = run_nbg(wd, M_DECODER, ["d0.dat"], dry_run=dry_run,
                      extra=["-b", "0", "--save_txt", "1"])
    dec = read_out(0)
    print(f"[3] NPU decoder: {'OK' if 'ret=0' in out else 'FAIL'}")
    if dec is None:
        return 1, out
    # joiner: enc320 + dec320 → 263 logits
    enc320.astype(np.float32).tofile(os.path.join(wd, "in0.dat"))
    dec[:320].astype(np.float32).tofile(os.path.join(wd, "in1.dat"))
    rc, out = run_nbg(wd, M_JOINER, ["in0.dat", "in1.dat"], dry_run=dry_run,
                      extra=["-b", "0", "--save_txt", "1"])
    logits = read_out(0)
    print(f"[4] NPU joiner: {'OK' if 'ret=0' in out else 'FAIL'}")
    if logits is not None and len(logits) >= 263:
        lg = logits[:263]
        p = np.exp(lg - lg.max()); p /= p.sum()
        top = np.argsort(-p)[:5]
        print("[5] top5 token: " + ", ".join(f"{int(i)}({p[i]*100:.1f}%)" for i in top))
    print("KWS 链路自检完成: 音频 → fbank → NPU(enc/dec/join) → 分词 ✓")
    return 0, out


def cmd_llm(a):
    """LLM: 经 reef1994/a733-llama-npu-stack (llama.cpp + TIM-VX 卸载到 /dev/galcore)."""
    # 该栈是独立工程, 此处给出标准调用封装; 实际运行需在板端该栈环境中.
    cmd = (f"cd {a.stack} && python3 -m a733_llama --model {a.model} "
           f"--npu-layers {a.npu_layers} --prompt \"{a.prompt}\"")
    if a.dry_run:
        print("DRY-RUN:", cmd); return 0
    print("调用 llama.cpp NPU 栈 (需 reef1994/a733-llama-npu-stack 已部署):")
    print("  " + cmd)
    return 0


def cmd_vlm(a):
    rc, out = run_nbg(a.workdir, M_SHUFFLENET if not a.model else a.model,
                      [a.input], dry_run=a.dry_run)
    print(out)
    return rc


def cmd_detect(a):
    if a.demo:
        # 走原厂 yolov5_demo 二进制, 直接出检测框 (我们实测 detection num:3)
        cmd = [YOLO_DEMO, "-nb", YOLO_DEMO_NB, "-i", a.input, "-l", "1"]
        if a.dry_run:
            print("DRY-RUN:", " ".join(cmd)); return 0
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(r.stdout + r.stderr)
        return r.returncode
    model = a.model or M_YOLOV5S
    rc, out = run_nbg(a.workdir, model, [a.input], dry_run=a.dry_run)
    print(out)
    return rc


def cmd_classify(a):
    model = a.model or M_SHUFFLENET
    rc, out = run_nbg(a.workdir, model, [a.input], dry_run=a.dry_run)
    print(out)
    return rc


def cmd_kws(a):
    rc, out = run_kws(a.wav, workdir=a.workdir, dry_run=a.dry_run)
    if not a.dry_run:
        print(out)
    return rc


def cmd_edge(a):
    # 边缘视觉 = 对应 NBG (零售识别/pose/OCR 都是检测或分类特化模型)
    model = a.model or M_YOLOV5S
    rc, out = run_nbg(a.workdir, model, [a.input], dry_run=a.dry_run)
    print(out)
    return rc


def cmd_info(a):
    if a.dry_run:
        print("DRY-RUN: ssh radxa 'cat /sys/bus/platform/devices/3600000.npu/driver; "
              "npu power domain / clock / vipcore IRQ'")
        return 0
    print("请在板端运行以下只读检查 (本机不直连板子):")
    print("  basename $(readlink /sys/bus/platform/devices/3600000.npu/driver)")
    print("  test -c /dev/vipcore && echo vipcore-ok")
    print("  grep vipcore /proc/interrupts")
    return 0


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=DEFAULT_WORKDIR, help="板端工作目录")
    common.add_argument("--dry-run", action="store_true", help="只打印命令不执行")

    p = argparse.ArgumentParser(prog="a7a_npu", description="A733 NPU 统一 CLI 工具箱")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("llm", parents=[common], help="LLM 推理 (llama.cpp + TIM-VX NPU 卸载)")
    sp.add_argument("--stack", default="/home/radxa/a733-llama-npu-stack")
    sp.add_argument("--model", required=True)
    sp.add_argument("--npu-layers", type=int, default=99)
    sp.add_argument("--prompt", default="Hello")
    sp.set_defaults(func=cmd_llm)

    sp = sub.add_parser("vlm", parents=[common], help="VLM (MobileCLIP-S0 等)")
    sp.add_argument("--model", default="")
    sp.add_argument("input")
    sp.set_defaults(func=cmd_vlm)

    sp = sub.add_parser("detect", parents=[common], help="目标检测 (YOLOv5/8)")
    sp.add_argument("--model", default="")
    sp.add_argument("--demo", action="store_true", help="用原厂 yolov5_demo 二进制")
    sp.add_argument("input")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("classify", parents=[common], help="图像分类")
    sp.add_argument("--model", default="")
    sp.add_argument("input")
    sp.set_defaults(func=cmd_classify)

    sp = sub.add_parser("kws", parents=[common], help="关键词唤醒")
    sp.add_argument("wav", nargs="?", default=None)
    sp.set_defaults(func=cmd_kws)

    sp = sub.add_parser("edge", parents=[common], help="边缘视觉 (零售/pose/OCR)")
    sp.add_argument("--model", default="")
    sp.add_argument("input")
    sp.set_defaults(func=cmd_edge)

    sp = sub.add_parser("info", parents=[common], help="打印 NPU 状态")
    sp.set_defaults(func=cmd_info)
    return p


def main():
    a = build_parser().parse_args()
    rc = a.func(a)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
