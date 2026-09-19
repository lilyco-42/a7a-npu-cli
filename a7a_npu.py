#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A733 / Radxa Cubie A7A hardware and NPU unified CLI.

This is a thin CLI adapter around the official Radxa/Allwinner tools. It does
not pretend to implement ACUITY, VIPLite, G2D, or model post-processing itself.
Commands fail clearly when the required official binary, model, repository, or
Docker image is missing.
"""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_WORKDIR = "/home/radxa/kws-min"
DEFAULT_VPM = "vpm_run"
DEFAULT_NBINFO = "nbinfo"
DEFAULT_ACUITY_IMAGE = "ubuntu-npu:v2.0.10.2"
MODEL_ZOO_URL = "https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz"

# The list is maintained from the official Radxa A7A Model Zoo index.
MODEL_ZOO = [
    ("yolo11", "YOLO11", "detection"),
    ("yolo11-seg", "YOLO11 Seg", "segmentation"),
    ("yolo11-pose", "YOLO11 Pose", "pose"),
    ("yolov8", "YOLOv8", "detection"),
    ("yolov8-seg", "YOLOv8 Seg", "segmentation"),
    ("yolov8-pose", "YOLOv8 Pose", "pose"),
    ("yolov3-darknet", "YOLOv3", "detection"),
    ("yolov5", "YOLOv5", "detection"),
    ("yolo26", "YOLO26", "detection"),
    ("yolox", "YOLOX", "detection"),
    ("retinaface", "RetinaFace", "face detection"),
    ("ppseg", "PPSeg", "segmentation"),
    ("mobilenetv1-tensorflow", "MobileNetV1", "classification"),
    ("mobilenetv2", "MobileNetV2", "classification"),
    ("resnet50-tflite", "ResNet50 TFLite", "classification"),
    ("resnet50v2", "ResNet50 V2", "classification"),
    ("densenet121-keras", "DenseNet121", "classification"),
    ("squeezenet-pytorch", "SqueezeNet", "classification"),
    ("lenet-caffe", "LeNet", "classification"),
    ("clip", "CLIP", "vision-language"),
    ("zipformer", "Zipformer", "ASR / KWS"),
    ("lite-transformer", "Lite Transformer", "ASR"),
    ("lstm", "LSTM", "sequence model"),
]

M_YOLOV5S = "models/yolov5s_rt_uint8_a733.nb"
M_SHUFFLENET = "models/official_a733.nb"
M_JOINER = "models/joiner_float_a733.nb"
M_DECODER = "models/decoder_float_a733.nb"
M_ENCODER = "models/encoder_float_a733.nb"
M_VOCODER = "models/vocoder_int16_a733.nb"
YOLO_DEMO = "/home/radxa/npu_zoo/examples/yolov5/build/yolov5_demo_a733"
YOLO_DEMO_NB = "/home/radxa/npu_zoo/examples/yolov5/model/yolov5s_rt_uint8_a733.nb"


def _command_text(cmd):
    return shlex.join([str(x) for x in cmd])


def _fail(message):
    print("ERROR:", message, file=sys.stderr)
    return 2


def _run(cmd, cwd=None, env=None, timeout=300, dry_run=False):
    cmd = [str(x) for x in cmd]
    if dry_run:
        print("DRY-RUN:", _command_text(cmd))
        return 0
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    except FileNotFoundError:
        return _fail("找不到可执行文件: " + cmd[0])
    except subprocess.TimeoutExpired:
        return _fail(f"命令超时({timeout}s): {_command_text(cmd)}")
    return p.returncode


def _run_capture(cmd, cwd=None, env=None, timeout=30):
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, text=True,
                              capture_output=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _env(workdir):
    lib = os.path.join(workdir, "lib")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    return dict(os.environ, LD_LIBRARY_PATH=f"{lib}:{old}")


def _resolve(path_or_name, workdir=None):
    if os.path.isabs(path_or_name):
        return path_or_name
    if workdir:
        candidate = os.path.join(workdir, path_or_name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which(path_or_name) or path_or_name


def _parse_irq():
    try:
        text = Path("/proc/interrupts").read_text(errors="replace")
    except OSError:
        return None
    rows = [line.strip() for line in text.splitlines()
            if "vipcore" in line.lower() or "npu" in line.lower()]
    return rows


def collect_info():
    driver_link = "/sys/bus/platform/devices/3600000.npu/driver"
    driver = os.path.basename(os.path.realpath(driver_link)) \
        if os.path.exists(driver_link) else None
    nodes = {}
    for node in ("/dev/vipcore", "/dev/galcore", "/dev/dri/renderD128",
                 "/dev/g2d", "/dev/dma_heap/system", "/dev/cedar_dev",
                 "/dev/cedar_dev_ve2"):
        nodes[node] = os.path.exists(node)

    def read_first(path):
        try:
            return Path(path).read_text(errors="replace").strip()
        except OSError:
            return None

    clock = None
    clk_summary = read_first("/sys/kernel/debug/clk/clk_summary")
    if clk_summary:
        clock = [line.strip() for line in clk_summary.splitlines()
                 if "npu" in line.lower()][:20]
    pm = read_first("/sys/kernel/debug/pm_genpd/pm_genpd_summary")
    power = [line.strip() for line in (pm or "").splitlines()
             if "npu" in line.lower()][:20]
    g2d_version = read_first("/sys/module/g2d_sunxi/version")
    return {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "driver": driver,
        "nodes": nodes,
        "g2d_module_version": g2d_version,
        "npu_clock_lines": clock,
        "npu_power_lines": power,
        "interrupt_lines": _parse_irq(),
        "debugfs_readable": bool(clk_summary or pm),
    }


def _collect_hardware():
    info = collect_info()
    def read(path):
        try:
            return Path(path).read_text(errors="replace").strip()
        except OSError:
            return None
    mem = read("/proc/meminfo")
    info["memory"] = {}
    if mem:
        for line in mem.splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                key, value = line.split(":", 1)
                info["memory"][key] = value.strip()
    info["temperatures"] = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        value = read(zone)
        type_path = zone.parent / "type"
        name = read(type_path) or zone.parent.name
        if value:
            try:
                info["temperatures"][name] = round(int(value) / 1000, 1)
            except ValueError:
                info["temperatures"][name] = value
    info["cpu_frequencies"] = {}
    for freq in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq")):
        value = read(freq)
        if value:
            info["cpu_frequencies"][freq.parent.parent.name] = value
    return info


def cmd_hardware(a):
    info = _collect_hardware()
    if a.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    print(f"kernel: {info['kernel']}  machine: {info['machine']}")
    print("accelerator/device nodes:")
    for node, present in info["nodes"].items():
        print(f"  {node}: {'present' if present else 'missing'}")
    print("memory:")
    for key, value in info["memory"].items():
        print(f"  {key}: {value}")
    print("temperatures:")
    for key, value in info["temperatures"].items():
        print(f"  {key}: {value} C")
    print("cpu frequencies:")
    for key, value in list(info["cpu_frequencies"].items())[:16]:
        print(f"  {key}: {value} kHz")
    return 0


def cmd_info(a):
    info = collect_info()
    if a.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"kernel: {info['kernel']}")
        print(f"machine: {info['machine']}")
        print(f"npu driver: {info['driver'] or 'missing'}")
        for node, present in info["nodes"].items():
            print(f"{node}: {'present' if present else 'missing'}")
        print(f"g2d module: {info['g2d_module_version'] or 'not loaded/unknown'}")
        if info["npu_clock_lines"]:
            print("npu clocks:")
            print("\n".join("  " + x for x in info["npu_clock_lines"]))
        else:
            print("npu clocks: unavailable (need root/debugfs)")
        if info["npu_power_lines"]:
            print("npu power domains:")
            print("\n".join("  " + x for x in info["npu_power_lines"]))
        print("interrupts:")
        print("\n".join("  " + x for x in (info["interrupt_lines"] or ["none"])))
    if a.strict:
        required = info["nodes"].get("/dev/vipcore", False)
        return 0 if info["driver"] == "vipcore" and required else 1
    return 0


def run_nbg(workdir, model_nb, inputs, loop=1, dry_run=False,
            extra=None, sample=None, vpm_binary=DEFAULT_VPM):
    """Run a prepared NBG through the official vpm_run executable.

    A sample file is generated only when --sample is not supplied. This is a
    deliberate write operation; use --sample to keep the configuration under
    version control and avoid silently creating files in a system directory.
    """
    workdir = os.path.abspath(workdir)
    if sample:
        sample_path = _resolve(sample, workdir)
    else:
        sample_path = os.path.join(workdir, "sample_run.txt")
        if not dry_run:
            try:
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write("[network]\n")
                    f.write(_resolve(model_nb, workdir) + "\n")
                    f.write("[input]\n")
                    for item in inputs:
                        f.write(_resolve(item, workdir) + "\n")
            except OSError as exc:
                return _fail(f"无法写入 sample 配置 {sample_path}: {exc}")
    cmd = [_resolve(vpm_binary, workdir), "-s", sample_path, "-l", str(loop)]
    if extra:
        cmd += extra
    if dry_run:
        print("DRY-RUN:", _command_text(cmd))
        return 0
    if not os.path.exists(cmd[0]) and not shutil.which(cmd[0]):
        return _fail(f"找不到 vpm_run: {cmd[0]}")
    try:
        p = subprocess.run(cmd, cwd=workdir, env=_env(workdir), timeout=300)
    except subprocess.TimeoutExpired:
        return _fail("vpm_run 超时；请检查 NBG、输入尺寸和 vipcore IRQ")
    return p.returncode


def cmd_vpm_run(a):
    if not a.sample and not a.model:
        return _fail("vpm run 需要 --sample，或同时提供 --model 和输入文件")
    if not a.sample and not a.inputs:
        return _fail("未提供输入文件；请使用 --sample 或在 --model 后给出 inputs")
    extra = []
    for flag, value in (
        ("-d", a.device), ("-t", a.timeout), ("-b", a.bypass),
        ("--show_top5", a.show_top5), ("--save_txt", a.save_txt),
        ("-c", a.core_index), ("--op_segment", a.op_segment),
        ("--layer_dump", a.layer_dump),
    ):
        if value is not None:
            extra += [flag, str(value)]
    if a.layer_profile_dump:
        extra += ["--layer_profile_dump", "1"]
    if a.preload:
        extra += ["--preload", "1"]
    return run_nbg(a.workdir, a.model or "", a.inputs, a.loop,
                   a.dry_run, extra, a.sample, a.binary)


def cmd_vpm_build(a):
    make = shutil.which("make") or "make"
    base = [make]
    if a.install:
        cmd = base + ["install", "AI_SDK_PLATFORM=a733",
                      f"INSTALL_PREFIX={a.prefix}"]
    else:
        cmd = base + ["AI_SDK_PLATFORM=a733"]
    return _run(cmd, cwd=a.source, dry_run=a.dry_run)


def cmd_nbinfo(a):
    flag = None
    for attr, option in (
        ("all", "-a"), ("brief", "-b"), ("header", "-n"),
        ("layers", "-l"), ("operations", "-o"), ("input", "-in"),
        ("output", "-out"), ("memory", "-m"),
    ):
        if getattr(a, attr):
            flag = option
            break
    if flag is None:
        flag = "-b"
    if a.detail_memory:
        cmd = [_resolve(a.binary), "-m", "-d", a.model]
    elif a.flash_memory:
        cmd = [_resolve(a.binary), "-m", "-f", a.model]
    else:
        cmd = [_resolve(a.binary), flag, a.model]
    return _run(cmd, dry_run=a.dry_run)


def _docker_inner(a, inner):
    workspace = os.path.abspath(a.workspace)
    return ["docker", "run", "--rm", "--ipc=host", "-v",
            f"{workspace}:/workspace", "-w", "/workspace", a.image,
            "bash", "-lc", inner]


def cmd_acuity_env(a):
    if a.action == "check":
        cmd = ["docker", "images", a.image]
    elif a.action == "load":
        if not a.tar:
            return _fail("acuity env load 需要 --tar")
        cmd = ["docker", "load", "-i", a.tar]
    else:
        cmd = ["docker", "run", "--ipc=host", "-it", "-v",
               f"{os.path.abspath(a.workspace)}:/workspace", a.image,
               "/bin/bash"]
    return _run(cmd, dry_run=a.dry_run)


def _acuity_command(a):
    if a.action == "simulate":
        project = shlex.quote(a.project)
        binary = shlex.quote(a.binary)
        data = shlex.quote(a.data)
        image = shlex.quote(a.input)
        ide_env = ""
        if a.use_ide:
            ide_env = (
                "export USE_IDE_LIB=1; "
                f"export VIVANTE_SDK_DIR={shlex.quote(a.ide_dir)}; "
                "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"
                f"{shlex.quote(a.ide_dir)}/../common/lib:{shlex.quote(a.ide_dir)}/lib; "
                "unset VSI_USE_IMAGE_PROCESS; "
            )
        inner = (
            f"cd {project} && {ide_env} make -f makefile.linux && "
            f"./{binary} {data} {image}"
        )
        return _docker_inner(a, inner)

    model_dir = a.model_dir.rstrip("/")
    name = os.path.basename(model_dir)
    script = a.script_dir.rstrip("/")
    if a.action == "import":
        inner = f"cd {shlex.quote(script)} && ./pegasus_import.sh {shlex.quote(model_dir)}"
    elif a.action == "quantize":
        if a.hybrid or a.kld or a.compute_entropy:
            model = f"{model_dir}/{name}.json"
            data = f"{model_dir}/{name}.data"
            meta = f"{model_dir}/{name}_inputmeta.yml"
            qfile = f"{model_dir}/{name}_{a.qtype}.quantize"
            pegasus = a.pegasus
            args = ["python3", pegasus, "quantize", "--model", model,
                    "--model-data", data, "--iterations", str(a.iterations),
                    "--device", "CPU", "--with-input-meta", meta]
            args += ["--hybrid" if a.hybrid else "--rebuild",
                     "--model-quantize", qfile,
                     "--quantizer", "perchannel_symmetric_affine" if a.qtype == "pcq" else "asymmetric_affine",
                     "--qtype", a.qtype]
            if a.compute_entropy:
                args += ["--compute-entropy"]
            if a.kld:
                args += ["--algorithm", "kl_divergence", "--batch-size", str(a.batch_size),
                         "--divergence-first-quantize-bits", str(a.kld_bits)]
                if a.mle:
                    args += ["--MLE"]
            inner = "cd /workspace && " + _command_text(args)
        else:
            inner = f"cd {shlex.quote(script)} && ./pegasus_quantize.sh {shlex.quote(model_dir)} {shlex.quote(a.qtype)} {a.iterations}"
    elif a.action == "infer":
        inner = f"cd {shlex.quote(script)} && ./pegasus_inference.sh {shlex.quote(model_dir)} {shlex.quote(a.qtype)}"
    elif a.action == "export":
        inner = f"cd {shlex.quote(script)} && ./pegasus_export_ovx.sh {shlex.quote(model_dir)} {shlex.quote(a.qtype)}"
    else:
        return None
    return _docker_inner(a, inner)


def cmd_acuity(a):
    if a.acuity_action == "quantize" and getattr(a, "qtype", None) == "float":
        return _fail("float 是不量化推理/导出模式；不能作为 pegasus_quantize 的 qtype")
    cmd = _acuity_command(a)
    if cmd is None:
        return _fail("未知 ACUITY 操作")
    if not a.dry_run and not shutil.which("docker"):
        return _fail("找不到 docker；ACUITY 官方流程必须在 X86 Docker 环境运行")
    return _run(cmd, dry_run=a.dry_run)


def cmd_model_zoo(a):
    if a.action == "list":
        if a.json:
            print(json.dumps([{"id": x, "name": y, "task": z,
                               "docs": f"https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo/{x}"}
                              for x, y, z in MODEL_ZOO], ensure_ascii=False, indent=2))
        else:
            for i, name, task in MODEL_ZOO:
                print(f"{i:24} {task:18} {name}")
        return 0
    if a.action == "download":
        url = a.url or MODEL_ZOO_URL
        print(f"download: {url}")
        if a.dry_run:
            return 0
        try:
            urllib.request.urlretrieve(url, a.output)
        except OSError as exc:
            return _fail(f"下载失败: {exc}")
        print(f"saved: {a.output}")
        return 0
    if a.action == "extract":
        if a.dry_run:
            print("DRY-RUN: extract", a.archive, "->", a.output)
            return 0
        try:
            with tarfile.open(a.archive, "r:gz") as tf:
                members = tf.getmembers()
                if a.pattern:
                    members = [m for m in members if a.pattern in m.name]
                root = os.path.abspath(a.output)
                for m in members:
                    target = os.path.abspath(os.path.join(root, m.name))
                    if not target.startswith(root + os.sep):
                        return _fail("拒绝解压路径穿越成员: " + m.name)
                tf.extractall(root, members=members)
                print(f"extracted: {len(members)} files")
        except (OSError, tarfile.TarError) as exc:
            return _fail(f"解压失败: {exc}")
        return 0
    return _fail("未知 Model Zoo 操作")


def _g2d_binary(op, examples):
    names = {
        "rotate": "g2d_rotation_or_mirror",
        "format": "g2d_format_conversion",
        "scale": "g2d_scaler_or_down_sampling",
        "fill": "g2d_color_fill",
    }
    return os.path.join(examples, names[op])


def cmd_g2d(a):
    if a.action == "doctor":
        info = collect_info()
        result = {
            "g2d": info["nodes"].get("/dev/g2d", False),
            "dma_heap_system": info["nodes"].get("/dev/dma_heap/system", False),
            "module_version": info["g2d_module_version"],
            "header": os.path.exists("/usr/include/bsp/linux/sunxi-g2d.h"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if a.json else
              "\n".join(f"{k}: {v}" for k, v in result.items()))
        return 0 if result["g2d"] and result["dma_heap_system"] else (1 if a.strict else 0)
    if a.action == "build":
        source_map = {
            "rotate": "g2d_rotation_or_mirror.c",
            "format": "g2d_format_conversion.c",
            "scale": "g2d_scaler_or_down_sampling.c",
            "fill": "g2d_color_fill.c",
        }
        src = os.path.join(a.source, source_map[a.operation])
        out = os.path.join(a.output, _g2d_binary(a.operation, ""))
        if not os.path.exists(src):
            return _fail(f"找不到官方 G2D 源码: {src}")
        os.makedirs(a.output, exist_ok=True)
        return _run([a.cc, src, "-o", out], dry_run=a.dry_run)
    if a.action == "run":
        binary = a.binary or _g2d_binary(a.operation, a.examples)
        if not os.path.exists(binary) and not a.dry_run:
            return _fail(f"找不到 G2D 示例二进制: {binary}；先运行 g2d build")
        program_args = list(a.args)
        if program_args[:1] == ["--"]:
            program_args = program_args[1:]
        return _run([binary] + program_args, dry_run=a.dry_run)
    return _fail("未知 G2D 操作")


def cmd_voice(a):
    # Preserve POSIX paths on the A7A; do not turn /home/... into D:\\home on a
    # Windows machine that is only generating a dry-run command.
    repo = a.repo
    if a.action == "matcha":
        script = os.path.join(repo, "tools", "matcha_npu.py")
        cmd = [sys.executable, script, "--text", a.text]
        if a.play:
            cmd.append("--play")
    else:
        script = os.path.join(repo, "assistant_fast.py")
        cmd = [sys.executable, script]
        if a.from_wav:
            cmd += ["--from-wav"] + a.from_wav
        if a.no_tts:
            cmd.append("--no-tts")
    if not a.dry_run and not os.path.exists(script):
        return _fail(f"找不到离线语音助手文件: {script}")
    return _run(cmd, cwd=repo, dry_run=a.dry_run)


def run_kws(wav=None, workdir=DEFAULT_WORKDIR, dry_run=False):
    """Run the official voice-assistant KWS assets.

    The previous prototype stopped after encoder/decoder and then passed
    nonexistent in0.dat/in1.dat to joiner. The official voice-assistant demo
    owns the streaming cache/token/post-processing state, so this CLI delegates
    the complete assistant path to it instead of claiming a partial KWS result.
    """
    repo = os.environ.get("A7A_VOICE_REPO", "/home/radxa/npu_demos/voice_assistant")
    script = os.path.join(repo, "assistant_fast.py")
    if not dry_run and not os.path.exists(script):
        return _fail(f"找不到官方离线语音助手: {script}；请先部署 voice-assistant")
    cmd = [sys.executable, script]
    if wav:
        cmd += ["--from-wav", wav]
    return _run(cmd, cwd=repo, dry_run=dry_run)


def cmd_legacy_task(a):
    if a.cmd == "llm":
        runner = shlex.split(a.runner)
        cmd = runner + ["--model", a.model, "--npu-layers", str(a.npu_layers), "--prompt", a.prompt]
        if not a.dry_run and not os.path.isdir(a.stack):
            return _fail(f"LLM 栈不存在: {a.stack}；先部署 reef1994/a733-llama-npu-stack")
        return _run(cmd, cwd=a.stack, dry_run=a.dry_run)
    if a.cmd == "detect":
        use_demo = a.demo or (not a.model and not a.raw)
        if use_demo:
            cmd = [YOLO_DEMO, "-nb", YOLO_DEMO_NB, "-i", a.input, "-l", "1"]
            if not a.dry_run and not os.path.exists(YOLO_DEMO):
                return _fail(f"找不到官方 YOLO demo: {YOLO_DEMO}")
            return _run(cmd, dry_run=a.dry_run)
        suffix = os.path.splitext(a.input.lower())[1]
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return _fail("raw NBG 路径不能直接吃图片；请使用 --demo，或传入 ACUITY/Vivante 生成的 .tensor/.dat")
        return run_nbg(a.workdir, a.model, [a.input], dry_run=a.dry_run)
    if a.cmd in ("classify", "vlm", "edge"):
        if not a.model:
            return _fail(f"{a.cmd} 是通用 NBG 原始运行器，必须显式指定 --model；它不会自动完成图像预处理/后处理")
        suffix = os.path.splitext(a.input.lower())[1]
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return _fail(f"{a.cmd} 的 vpm_run 输入必须是 ACUITY/Vivante 生成的 .tensor/.dat；图片请先做预处理，或使用对应官方 demo")
        extra = ["-b", "0", "--show_top5", "1"] if a.cmd == "classify" else None
        return run_nbg(a.workdir, a.model, [a.input], dry_run=a.dry_run, extra=extra)
    if a.cmd == "kws":
        return run_kws(a.wav, a.workdir, a.dry_run)
    return _fail("未知任务")


def cmd_capabilities(a):
    rows = [
        ("A7A 硬件资源只读盘点", "hardware --json", "CLI wrapper; no writes"),
        ("Acuity 环境/Docker", "acuity env", "CLI wrapper; needs official A733 image"),
        ("模型导入", "acuity import", "CLI wrapper; needs ACUITY image"),
        ("量化 float/uint8/int16/pcq/bf16", "acuity quantize", "CLI wrapper"),
        ("KLD/MLE/熵/混合量化", "acuity quantize --kld/--mle/--hybrid --compute-entropy", "CLI wrapper"),
        ("量化/float 推理", "acuity infer", "CLI wrapper"),
        ("NBG/OpenVX 导出", "acuity export", "CLI wrapper"),
        ("板端 vpm_run 全参数", "vpm run", "CLI wrapper"),
        ("NBG 结构/层/算子/内存分析", "nbinfo", "CLI wrapper; needs nbinfo"),
        ("Model Zoo 23 项目目录/下载/解压", "model-zoo", "generic catalog; demos need their assets"),
        ("YOLOv5 检测框", "detect", "underlying board path verified: 3 boxes"),
        ("离线 KWS/ASR/TTS", "voice / kws", "voice wrapper; full audio assets required"),
        ("G2D rotate/format/scale/fill", "g2d", "wrapper around official C examples"),
        ("TIM-VX runtime", "llm", "runner wrapper; not a general LLM NPU guarantee"),
    ]
    if a.json:
        print(json.dumps([{"official": x, "cli": y, "status": z} for x, y, z in rows], ensure_ascii=False, indent=2))
    else:
        print("官方文档工具级 CLI 覆盖清单（不是每个 Model Zoo 后处理器都内置）")
        for official, cli, status in rows:
            print(f"{official:26} {cli:58} {status}")
    return 0


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=DEFAULT_WORKDIR, help="板端 NPU 工作目录")
    common.add_argument("--dry-run", action="store_true", help="只打印命令，不执行、不下载、不写文件")

    p = argparse.ArgumentParser(prog="a7a_npu", description="A733 / Radxa A7A 硬件与 NPU 统一 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("capabilities", help="显示官方文档功能与 CLI 覆盖矩阵")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_capabilities)

    sp = sub.add_parser("info", parents=[common], help="只读诊断 NPU/G2D/媒体设备")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--strict", action="store_true")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("hardware", parents=[common], help="只读盘点 A7A 硬件资源/温度/内存/设备节点")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_hardware)

    sp = sub.add_parser("vpm", help="官方 vpm_run 编译与运行")
    vsub = sp.add_subparsers(dest="vpm_action", required=True)
    run = vsub.add_parser("run", parents=[common], help="运行 sample.txt 中的 NBG")
    run.add_argument("--sample", help="已有 sample.txt；不提供则生成 sample_run.txt")
    run.add_argument("--model", default="", help="未提供 --sample 时的 NBG")
    run.add_argument("inputs", nargs="*", help="未提供 --sample 时的输入文件")
    run.add_argument("--binary", default=DEFAULT_VPM)
    run.add_argument("--loop", type=int, default=1)
    run.add_argument("--device", type=int)
    run.add_argument("--timeout", type=int)
    run.add_argument("--bypass", type=int)
    run.add_argument("--show-top5", dest="show_top5", type=int)
    run.add_argument("--save-txt", dest="save_txt", type=int)
    run.add_argument("--core-index", dest="core_index", type=int)
    run.add_argument("--layer-profile-dump", action="store_true")
    run.add_argument("--preload", action="store_true")
    run.add_argument("--op-segment")
    run.add_argument("--layer-dump")
    run.set_defaults(func=cmd_vpm_run)
    build = vsub.add_parser("build", parents=[common], help="make / make install AI_SDK_PLATFORM=a733")
    build.add_argument("--source", default=".")
    build.add_argument("--install", action="store_true")
    build.add_argument("--prefix", default="./")
    build.set_defaults(func=cmd_vpm_build)

    sp = sub.add_parser("nbinfo", parents=[common], help="官方 NBG 分析工具封装")
    sp.add_argument("model")
    sp.add_argument("--binary", default=DEFAULT_NBINFO)
    for name, text in (("all", "-a"), ("brief", "-b"), ("header", "-n"),
                       ("layers", "-l"), ("operations", "-o"), ("input", "-in"),
                       ("output", "-out"), ("memory", "-m")):
        sp.add_argument("--" + name, action="store_true", help=text)
    sp.add_argument("--detail-memory", action="store_true")
    sp.add_argument("--flash-memory", action="store_true")
    sp.set_defaults(func=cmd_nbinfo)

    sp = sub.add_parser("acuity", help="官方 ACUITY Docker 流程封装")
    asub = sp.add_subparsers(dest="acuity_action", required=True)
    env = asub.add_parser("env", parents=[common], help="检查/载入/进入 ACUITY Docker")
    env.add_argument("action", choices=["check", "load", "shell"])
    env.add_argument("--image", default=DEFAULT_ACUITY_IMAGE)
    env.add_argument("--tar")
    env.add_argument("--workspace", default=".")
    env.set_defaults(func=cmd_acuity_env)
    for action in ("import", "quantize", "infer", "export"):
        ap = asub.add_parser(action, parents=[common])
        ap.add_argument("model_dir")
        ap.add_argument("--workspace", default=".")
        ap.add_argument("--script-dir", default=".")
        ap.add_argument("--image", default=DEFAULT_ACUITY_IMAGE)
        ap.add_argument("--qtype", default="int16", choices=["float", "uint8", "int16", "pcq", "bf16"])
        ap.add_argument("--iterations", type=int, default=10)
        ap.add_argument("--pegasus", default="/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py")
        ap.add_argument("--kld", action="store_true")
        ap.add_argument("--mle", action="store_true")
        ap.add_argument("--kld-bits", type=int, default=12)
        ap.add_argument("--batch-size", type=int, default=100)
        ap.add_argument("--hybrid", action="store_true")
        ap.add_argument("--compute-entropy", action="store_true")
        ap.set_defaults(acuity_action=action, action=action, func=cmd_acuity)
    sim = asub.add_parser("simulate", parents=[common], help="Vivante IDE/OpenVX/NBG PC 侧模拟")
    sim.add_argument("--project", required=True, help="wksp 下的 OpenVX/NBG 项目目录")
    sim.add_argument("--binary", required=True, help="make 生成的项目可执行文件名")
    sim.add_argument("--data", required=True, help="export.data 或 network_binary.nb")
    sim.add_argument("--input", required=True, help="图片/输入文件")
    sim.add_argument("--workspace", default=".")
    sim.add_argument("--image", default=DEFAULT_ACUITY_IMAGE)
    sim.add_argument("--use-ide", action="store_true")
    sim.add_argument("--ide-dir", default="/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator")
    sim.set_defaults(acuity_action="simulate", action="simulate", func=cmd_acuity)

    sp = sub.add_parser("model-zoo", help="官方 Model Zoo 目录/下载/解压")
    zsub = sp.add_subparsers(dest="zoo_action", required=True)
    zl = zsub.add_parser("list")
    zl.add_argument("--json", action="store_true")
    zl.set_defaults(action="list", func=cmd_model_zoo)
    zd = zsub.add_parser("download", parents=[common])
    zd.add_argument("--url", default=MODEL_ZOO_URL)
    zd.add_argument("--output", default="allwinner-model-zoo.tar.gz")
    zd.set_defaults(action="download", func=cmd_model_zoo)
    ze = zsub.add_parser("extract", parents=[common])
    ze.add_argument("archive")
    ze.add_argument("--output", default=".")
    ze.add_argument("--pattern")
    ze.set_defaults(action="extract", func=cmd_model_zoo)

    sp = sub.add_parser("g2d", help="官方 G2D 示例的 CLI 封装")
    gsub = sp.add_subparsers(dest="g2d_action", required=True)
    gd = gsub.add_parser("doctor", parents=[common])
    gd.add_argument("--json", action="store_true")
    gd.add_argument("--strict", action="store_true")
    gd.set_defaults(action="doctor", func=cmd_g2d)
    gb = gsub.add_parser("build", parents=[common])
    gb.add_argument("operation", choices=["rotate", "format", "scale", "fill"])
    gb.add_argument("--source", required=True)
    gb.add_argument("--output", default="./bin")
    gb.add_argument("--cc", default="gcc")
    gb.set_defaults(action="build", func=cmd_g2d)
    gr = gsub.add_parser("run", parents=[common])
    gr.add_argument("operation", choices=["rotate", "format", "scale", "fill"])
    gr.add_argument("--examples", default="/home/radxa/g2d/bin")
    gr.add_argument("--binary")
    gr.add_argument("args", nargs=argparse.REMAINDER)
    gr.set_defaults(action="run", func=cmd_g2d)

    sp = sub.add_parser("voice", help="官方离线语音助手 CLI 封装")
    vsub = sp.add_subparsers(dest="voice_action", required=True)
    va = vsub.add_parser("assistant", parents=[common])
    va.add_argument("--repo", default="/home/radxa/npu_demos/voice_assistant")
    va.add_argument("--from-wav", nargs="*")
    va.add_argument("--no-tts", action="store_true")
    va.set_defaults(action="assistant", func=cmd_voice)
    vm = vsub.add_parser("matcha", parents=[common])
    vm.add_argument("--repo", default="/home/radxa/npu_demos/voice_assistant")
    vm.add_argument("--text", required=True)
    vm.add_argument("--play", action="store_true")
    vm.set_defaults(action="matcha", func=cmd_voice)

    # Task-oriented shortcuts. They remain, but their boundaries are explicit.
    sp = sub.add_parser("llm", parents=[common], help="LLM runner wrapper; not general NPU support")
    sp.add_argument("--stack", default="/home/radxa/a733-llama-npu-stack")
    sp.add_argument("--model", required=True)
    sp.add_argument("--npu-layers", type=int, default=99)
    sp.add_argument("--prompt", default="Hello")
    sp.add_argument("--runner", default="python3 -m a733_llama")
    sp.set_defaults(func=cmd_legacy_task)
    for name, help_text in (("vlm", "raw VLM NBG runner"), ("classify", "classification NBG runner"),
                            ("edge", "raw edge-vision NBG runner")):
        sp = sub.add_parser(name, parents=[common], help=help_text)
        sp.add_argument("--model", default="")
        sp.add_argument("input")
        sp.set_defaults(func=cmd_legacy_task)
    sp = sub.add_parser("detect", parents=[common], help="YOLOv5 official demo or raw NBG")
    sp.add_argument("--model", default="")
    sp.add_argument("--demo", action="store_true")
    sp.add_argument("--raw", action="store_true")
    sp.add_argument("input")
    sp.set_defaults(func=cmd_legacy_task)
    sp = sub.add_parser("kws", parents=[common], help="official offline voice-assistant KWS shortcut")
    sp.add_argument("wav", nargs="?", default=None)
    sp.add_argument("--from-wav", dest="wav", help="official assistant wav input")
    sp.set_defaults(func=cmd_legacy_task)

    return p


def main():
    args = build_parser().parse_args()
    rc = args.func(args)
    raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
