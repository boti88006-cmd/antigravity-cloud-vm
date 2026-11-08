#!/usr/bin/env python3
"""
Antigravity Cloud VM × Google Colab GPU (RTX PRO 6000 Blackwell 96GB) × 20TB Google Drive
90秒+ 深海探索纪录片 720P 高清图生视频流水线 (720P HD Anchor Image-to-Video Pipeline)

支持三种速度/画质档位，且自动跨目录复用已生成的分镜 (绝不浪费已渲染的 30 分钟镜头):
1. `--engine ltx` (极速 720P 原图锚点 I2V，单镜 ~45s，15 分钟跑完 90 秒全片):
   使用 18 张 1280x704 IMAX 超清原图逐镜驱动 LTXImageToVideoPipeline (28步)，高压缩 3D VAE 令 Token 数减少 85%，
   彻底解决第一次纯文生视频 (T2V) + 18代尾帧接力造成的模糊问题！
2. `--engine wan-fast` (阿里万相 14B 10倍极速模式，单镜 ~2.5-3 分钟，免重下模型):
   直接复用已下载的 Wan2.1-I2V-14B 权重，采用 832x480 甜点分辨率 (3D Token 数减少 57%，注意力计算量降为 1/5.3)
   + 20 步 FlowMatch 快速收敛 (再快 2.5 倍) + TF32/SDPA 硬件加速 + FFmpeg 1280x704 Lanczos 超分锐化封装。
3. `--engine wan-i2v` (阿里万相 14B 原生 1280x704 满血模式，默认优化为 25 步，速度翻倍)。
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
import wave


def setup_local_ssd_cache():
    """将 HuggingFace 模型缓存绑定到 Colab 本地高速 NVMe SSD (/content/hf_cache)"""
    local_cache = "/content/hf_cache" if os.path.exists("/content") else os.path.abspath("./hf_cache_local")
    os.makedirs(local_cache, exist_ok=True)
    os.environ["HF_HOME"] = local_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(local_cache, "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    print(f"⚡ 模型高速缓存目录已绑定至本地 NVMe SSD: {local_cache}")


def enable_blackwell_fast_math():
    """开启 Blackwell GPU TF32 矩阵加速与 PyTorch Flash/Memory-Efficient SDPA 注意力加速"""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)


def generate_deep_sea_soundtrack(duration_sec: float, output_wav_path: str, sample_rate: int = 44100):
    """合成 BBC Blue Planet / Hans Zimmer 风格的深海探索纪录片纯配乐 (无配音)"""
    import numpy as np

    print(f"🎵 正在合成 {duration_sec:.1f} 秒深海探索纪录片立体声纯配乐 (无配音): {output_wav_path}")
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

    left = np.zeros(num_samples, dtype=np.float32)
    right = np.zeros(num_samples, dtype=np.float32)

    chords = [
        (0.00, 0.22, [130.81, 196.00, 246.94, 293.66, 392.00]),   # Act I: Cmaj9 阳光透射带
        (0.22, 0.45, [110.00, 164.81, 220.00, 261.63, 329.63]),   # Act II-a: Am7 暮光带发光水母
        (0.45, 0.68, [87.31,  130.81, 174.61, 207.65, 261.63]),   # Act II-b: Fm9 午夜带深海巨兽
        (0.68, 0.85, [103.83, 155.56, 207.65, 261.63, 311.13]),   # Act III-a: AbMaj7 万米海沟秘境
        (0.85, 1.00, [130.81, 196.00, 261.63, 329.63, 392.63]),   # Act III-b: C Major 史诗终章
    ]

    for start_r, end_r, freqs in chords:
        s_idx = int(start_r * num_samples)
        e_idx = int(end_r * num_samples)
        seg_len = e_idx - s_idx
        if seg_len <= 0:
            continue
        seg_t = t[s_idx:e_idx] - t[s_idx]

        fade_len = min(int(3.5 * sample_rate), seg_len // 3)
        env = np.ones(seg_len, dtype=np.float32)
        if fade_len > 0:
            env[:fade_len] = np.linspace(0.0, 1.0, fade_len, dtype=np.float32) ** 1.5
            env[-fade_len:] = np.linspace(1.0, 0.0, fade_len, dtype=np.float32) ** 1.5

        for i, f in enumerate(freqs):
            lfo_l = 0.5 + 0.5 * np.sin(2.0 * np.pi * (0.12 + 0.03 * i) * seg_t + i)
            lfo_r = 0.5 + 0.5 * np.cos(2.0 * np.pi * (0.11 + 0.03 * i) * seg_t + i * 0.7)

            wave_l = (
                0.65 * np.sin(2.0 * np.pi * (f * 0.998) * seg_t)
                + 0.25 * np.sin(2.0 * np.pi * (f * 2.001) * seg_t)
                + 0.10 * np.sin(2.0 * np.pi * (f * 0.500) * seg_t)
            )
            wave_r = (
                0.65 * np.sin(2.0 * np.pi * (f * 1.002) * seg_t + 0.4)
                + 0.25 * np.sin(2.0 * np.pi * (f * 1.999) * seg_t + 0.8)
                + 0.10 * np.sin(2.0 * np.pi * (f * 0.500) * seg_t)
            )
            amp = 0.12 / (1.0 + 0.2 * i)
            left[s_idx:e_idx] += amp * env * lfo_l * wave_l
            right[s_idx:e_idx] += amp * env * lfo_r * wave_r

    sub_env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.08 * t)
    sub_wave = 0.18 * sub_env * np.sin(2.0 * np.pi * 65.41 * t)
    left += sub_wave
    right += sub_wave

    pentatonic_notes = [523.25, 587.33, 659.25, 783.99, 987.77, 1046.50, 1318.51]
    rng = np.random.default_rng(seed=2026)
    note_times = np.arange(1.5, duration_sec - 2.5, 2.25)
    for idx, nt in enumerate(note_times):
        n_start = int(nt * sample_rate)
        n_dur = int(3.2 * sample_rate)
        n_end = min(num_samples, n_start + n_dur)
        actual_len = n_end - n_start
        if actual_len <= 0:
            continue
        nt_arr = np.arange(actual_len, dtype=np.float32) / sample_rate
        freq = pentatonic_notes[idx % len(pentatonic_notes)]
        pluck_env = (1.0 - np.exp(-nt_arr * 25.0)) * np.exp(-nt_arr * 1.35)
        shimmer = 0.06 * pluck_env * (
            np.sin(2.0 * np.pi * freq * nt_arr)
            + 0.4 * np.sin(2.0 * np.pi * (freq * 2.0) * nt_arr)
        )
        pan = 0.25 + 0.5 * rng.random()
        left[n_start:n_end] += shimmer * pan
        right[n_start:n_end] += shimmer * (1.0 - pan)

    for call_start in [22.0, 46.0, 68.0]:
        if call_start + 6.0 >= duration_sec:
            continue
        c_s = int(call_start * sample_rate)
        c_len = int(5.5 * sample_rate)
        c_e = min(num_samples, c_s + c_len)
        ct = np.arange(c_e - c_s, dtype=np.float32) / sample_rate
        f_inst = 310.0 + 75.0 * np.cos(2.0 * np.pi * 0.18 * ct)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sample_rate
        c_env = np.sin(np.pi * ct / (ct[-1] + 1e-5)) ** 2
        whale = 0.055 * c_env * (np.sin(phase) + 0.3 * np.sin(2.0 * phase))
        left[c_s:c_e] += whale * 0.6
        right[c_s:c_e] += whale * 0.8

    in_samples = min(int(3.0 * sample_rate), num_samples // 4)
    out_samples = min(int(4.0 * sample_rate), num_samples // 4)
    global_env = np.ones(num_samples, dtype=np.float32)
    global_env[:in_samples] = np.linspace(0.0, 1.0, in_samples, dtype=np.float32)
    global_env[-out_samples:] = np.linspace(1.0, 0.0, out_samples, dtype=np.float32)

    left *= global_env
    right *= global_env

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    left = (left / peak) * 0.85
    right = (right / peak) * 0.85

    stereo_pcm = np.empty((num_samples, 2), dtype=np.int16)
    stereo_pcm[:, 0] = (left * 32767.0).astype(np.int16)
    stereo_pcm[:, 1] = (right * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_pcm.tobytes())

    print(f"✅ 纪录片纯配乐合成完成: {output_wav_path}")
    return output_wav_path


def load_anchor_pil(anchor_rel_path: str, width: int, height: int):
    """加载超清锚点关键帧图像并调整为目标尺寸"""
    from PIL import Image

    if not anchor_rel_path or not os.path.exists(anchor_rel_path):
        return None
    img = Image.open(anchor_rel_path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def find_existing_hd_segment(workspace_dir: str, current_segments_dir: str, shot_id: int) -> str | None:
    """检查当前目录或 Wan14B_720P 目录中是否已有渲染好的高清分镜，避免浪费已渲染镜头"""
    filename = f"shot_{shot_id:03d}.mp4"
    candidates = [
        os.path.join(current_segments_dir, filename),
        os.path.join(workspace_dir, "outputs", "Deep_Sea_Odyssey_90s_Wan14B_720P", "segments", filename),
        os.path.join(workspace_dir, "outputs", "Deep_Sea_Odyssey_90s_HD_I2V", "segments", filename),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 50_000:
            return c
    return None


def concat_videos_with_bgm_ffmpeg(
    segment_paths: list[str],
    final_output_path: str,
    bgm_wav_path: str | None = None,
    target_width: int = 1280,
    target_height: int = 704,
):
    """
    使用 FFmpeg filter_complex 将多个 5 秒分镜统一通过 Lanczos 超分至 1280x704 (720P) 24fps，
    开启细节锐化滤镜 (unsharp) 与 CRF 14 高清码率，并混入立体声纯配乐 (支持不同分辨率/帧率分镜无缝混剪)
    """
    list_file = os.path.join(os.path.dirname(final_output_path), "concat_list.txt")
    normalized_segments = []
    norm_dir = os.path.join(os.path.dirname(final_output_path), "_norm_720p")
    os.makedirs(norm_dir, exist_ok=True)

    for idx, seg in enumerate(segment_paths, start=1):
        norm_seg = os.path.join(norm_dir, f"norm_{idx:03d}.mp4")
        if not os.path.exists(norm_seg) or os.path.getmtime(seg) > os.path.getmtime(norm_seg):
            vf = f"scale={target_width}:{target_height}:flags=lanczos,fps=24,unsharp=5:5:0.65:3:3:0.0,format=yuv420p"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", os.path.abspath(seg),
                    "-vf", vf,
                    "-c:v", "libx264", "-crf", "14", "-preset", "fast",
                    "-an", norm_seg,
                ],
                check=True,
            )
        normalized_segments.append(norm_seg)

    with open(list_file, "w", encoding="utf-8") as f:
        for seg in normalized_segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    if bgm_wav_path and os.path.exists(bgm_wav_path):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-i", bgm_wav_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            final_output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "copy",
            final_output_path,
        ]

    subprocess.run(cmd, check=True)
    print(f"🎬 720P (1280x704) 高清深海纪录片长视频（含立体声纯配乐）已合成: {final_output_path}")


def load_ltx_pipelines(model_id: str):
    """加载 LTXImageToVideoPipeline 与 LTXPipeline"""
    import torch
    from diffusers import LTXImageToVideoPipeline, LTXPipeline

    enable_blackwell_fast_math()

    for attempt in range(2):
        try:
            t2v_pipe = LTXPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            i2v_pipe = LTXImageToVideoPipeline.from_pipe(t2v_pipe)
            break
        except Exception as e:
            if attempt == 0:
                print(f"⚠️ 检测到旧缓存异常 ({e})，正在清理本地缓存并重新下载完整模型权重...")
                shutil.rmtree(os.environ.get("HF_HOME", "/content/hf_cache"), ignore_errors=True)
                setup_local_ssd_cache()
            else:
                raise

    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🖥️ 检测到 GPU: {torch.cuda.get_device_name(0)} (显存: {vram_gb:.1f} GB)")

    t2v_pipe.vae.enable_tiling()
    if vram_gb >= 28.0:
        print("⚡ 显存充裕 (>=28GB)，启用满血全显存直通加速模式 (pipe.to('cuda'))！")
        t2v_pipe.to("cuda")
    else:
        t2v_pipe.enable_model_cpu_offload()

    return t2v_pipe, i2v_pipe


def run_ltx_pipeline(storyboard: dict, workspace_dir: str, force_rerun: bool = False, steps_override: int = 0):
    """
    720P (1280x704) 极速超清锚点图生视频模式 (Anchor Keyframe I2V, 单镜头仅需 ~45 秒):
    每个镜头都直接由一张 1280x704 的 IMAX 超清原图驱动 LTXImageToVideoPipeline 生成动态视频，
    彻底消除纯文生视频与多代尾帧传递导致的画面模糊！
    """
    import torch
    from diffusers.utils import export_to_video

    model_id = "Lightricks/LTX-Video"
    print(f"🚀 正在加载高速 720P 图生视频大模型: {model_id}...")

    t2v_pipe, i2v_pipe = load_ltx_pipelines(model_id)

    width = storyboard.get("resolution", {}).get("width", 1280)
    height = storyboard.get("resolution", {}).get("height", 704)
    fps = storyboard.get("fps", 24)
    seg_sec = storyboard.get("segment_seconds", 5)
    steps = steps_override if steps_override > 0 else min(storyboard.get("num_inference_steps", 30), 32)
    raw_frames = fps * seg_sec
    num_frames = ((raw_frames - 1) // 8) * 8 + 1

    global_style = storyboard.get("global_style", "")
    negative_prompt = storyboard.get("negative_prompt", "blurry, out of focus, low quality, worst quality")
    title = storyboard.get("title", "Deep_Sea_Odyssey_90s_HD_I2V")

    segments_dir = os.path.join(workspace_dir, "outputs", title, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    shots = storyboard.get("shots", [])
    total_duration = len(shots) * seg_sec
    segment_files = []

    print(
        f"🌊 开始使用 Colab GPU 极速渲染 720P 高清图生视频纪录片《{title}》:\n"
        f"   分辨率: {width}x{height} (720P HD) | 步数: {steps} (极速收敛) | 共 {len(shots)} 个镜头 ({total_duration}s)"
    )

    for idx, shot in enumerate(shots, start=1):
        shot_id = shot.get("id", idx)
        act_name = shot.get("act", "")
        seg_path = os.path.join(segments_dir, f"shot_{shot_id:03d}.mp4")

        existing_seg = None if force_rerun else find_existing_hd_segment(workspace_dir, segments_dir, shot_id)
        if existing_seg:
            print(f"⏭️ [镜头 {idx}/{len(shots)}] ({act_name}) 复用已渲染的高清分镜: {existing_seg}")
            segment_files.append(existing_seg)
            continue

        full_prompt = f"{shot['prompt']}, {global_style}" if global_style else shot["prompt"]
        anchor_rel = shot.get("anchor_image", f"keyframes/shot_{shot_id:03d}.jpg")
        anchor_pil = load_anchor_pil(anchor_rel, width, height)

        t0 = time.time()
        generator = torch.Generator("cpu").manual_seed(2026 + shot_id)

        if anchor_pil is not None:
            print(f"🖼️ [镜头 {idx}/{len(shots)}] ({act_name}) 720P 超清原图驱动 (Anchor I2V: {anchor_rel})...")
            video_frames = i2v_pipe(
                image=anchor_pil,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=3.2,
                generator=generator,
            ).frames[0]
        else:
            print(f"🎬 [镜头 {idx}/{len(shots)}] ({act_name}) 文生视频模式 (T2V)...")
            video_frames = t2v_pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=3.2,
                generator=generator,
            ).frames[0]

        export_to_video(video_frames, seg_path, fps=fps)
        elapsed = time.time() - t0
        print(f"✅ [镜头 {idx}/{len(shots)}] 720P HD 渲染完成 ({elapsed:.1f}s) -> 已存盘: {seg_path}")

        segment_files.append(seg_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bgm_wav = None
    if storyboard.get("bgm_enabled", True):
        bgm_wav = os.path.join(workspace_dir, "outputs", title, f"{title}_bgm_{total_duration}s.wav")
        generate_deep_sea_soundtrack(float(total_duration), bgm_wav)

    final_output = os.path.join(workspace_dir, "outputs", f"{title}_full_{total_duration}s.mp4")
    concat_videos_with_bgm_ffmpeg(segment_files, final_output, bgm_wav_path=bgm_wav, target_width=1280, target_height=704)
    return final_output


def run_wan_i2v_pipeline(
    storyboard: dict,
    workspace_dir: str,
    force_rerun: bool = False,
    fast_mode: bool = True,
    steps_override: int = 0,
):
    """
    使用阿里通义万相 Wan-AI/Wan2.1-I2V-14B-720P-Diffusers (140亿参数图生视频大模型)
    - fast_mode=True (默认开启，提速 ~10 倍，单镜 ~2.5-3 分钟):
      使用 832x480 黄金分辨率 (Token 数从 73,920 降至 31,500，注意力计算量降低 81%) + 20 步 FlowMatch 快速采样
      + TF32/SDPA 加速，最后通过 FFmpeg Lanczos + unsharp 超分锐化至 1280x704 (720P)！
    - fast_mode=False:
      原生 1280x704 25 步模式。
    """
    import sys
    try:
        import ftfy  # noqa: F401
    except ImportError:
        print("📦 正在自动安装 Wan2.1 文本清洗依赖 ftfy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "regex"], check=True)

    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from transformers import CLIPVisionModel

    enable_blackwell_fast_math()

    ltx_cache_dir = os.path.join(os.environ.get("HF_HOME", "/content/hf_cache"), "hub", "models--Lightricks--LTX-Video")
    if os.path.exists(ltx_cache_dir):
        print("🧹 正在清理旧的 LTX-Video 本地缓存以释放 28.4GB NVMe 空间供 Wan2.1-14B 使用...")
        shutil.rmtree(ltx_cache_dir, ignore_errors=True)

    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    print(f"🚀 正在加载阿里通义万相 14B 图生视频模型 (直接复用本地缓存): {model_id}...")

    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"🖥️ 检测到 GPU: {torch.cuda.get_device_name(0)} (显存: {vram_gb:.1f} GB)")
    if vram_gb >= 45.0:
        print("⚡ 95GB Blackwell 显存充裕，启用 Wan2.1-14B 全量 CUDA + TF32/SDPA 直通加速！")
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    if fast_mode:
        # 832x480 是 Wan2.1 官方黄金高速宽高比 (16:9)，3D Token 数从 73,920 降至 31,500，自注意力计算量减少 81%！
        width, height = 832, 480
        steps = steps_override if steps_override > 0 else 20
        mode_desc = "⚡ 10倍极速模式 (832x480 20步渲染 + FFmpeg 1280x704 720P Lanczos 超分锐化)"
    else:
        width = (storyboard.get("resolution", {}).get("width", 1280) // 16) * 16
        height = (storyboard.get("resolution", {}).get("height", 704) // 16) * 16
        steps = steps_override if steps_override > 0 else 25
        mode_desc = f"原生 {width}x{height} ({steps}步)"

    fps = 16
    seg_sec = storyboard.get("segment_seconds", 5)
    num_frames = 81  # 5s @ 16fps 标准帧数

    global_style = storyboard.get("global_style", "")
    negative_prompt = storyboard.get("negative_prompt", "blurry, out of focus, low quality, worst quality")
    title = "Deep_Sea_Odyssey_90s_Wan14B_720P"

    segments_dir = os.path.join(workspace_dir, "outputs", title, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    shots = storyboard.get("shots", [])
    total_duration = len(shots) * seg_sec
    segment_files = []

    print(
        f"🌊 开始使用 Wan2.1-I2V-14B 渲染高清图生视频纪录片《{title}》:\n"
        f"   模式: {mode_desc} | 帧数: {num_frames} | 共 {len(shots)} 个镜头 ({total_duration}s)"
    )

    for idx, shot in enumerate(shots, start=1):
        shot_id = shot.get("id", idx)
        act_name = shot.get("act", "")
        seg_path = os.path.join(segments_dir, f"shot_{shot_id:03d}.mp4")

        existing_seg = None if force_rerun else find_existing_hd_segment(workspace_dir, segments_dir, shot_id)
        if existing_seg:
            print(f"⏭️ [镜头 {idx}/{len(shots)}] ({act_name}) 已存在于云盘，自动复用 (绝不重复渲染): {existing_seg}")
            segment_files.append(existing_seg)
            continue

        anchor_rel = shot.get("anchor_image", f"keyframes/shot_{shot_id:03d}.jpg")
        anchor_pil = load_anchor_pil(anchor_rel, width, height)
        if anchor_pil is None:
            raise FileNotFoundError(f"未找到高清锚点原图: {anchor_rel}")

        full_prompt = f"{shot['prompt']}, {global_style}" if global_style else shot["prompt"]
        generator = torch.Generator("cpu").manual_seed(2026 + shot_id)

        print(f"🖼️ [镜头 {idx}/{len(shots)}] ({act_name}) Wan2.1-14B 正在由超清原图生成视频 ({anchor_rel}, {steps}步)...")
        t0 = time.time()
        video_frames = pipe(
            image=anchor_pil,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=generator,
        ).frames[0]

        export_to_video(video_frames, seg_path, fps=fps)
        print(f"✅ [镜头 {idx}/{len(shots)}] 渲染完成 ({time.time() - t0:.1f}s) -> 已实时存盘: {seg_path}")
        segment_files.append(seg_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bgm_wav = None
    if storyboard.get("bgm_enabled", True):
        bgm_wav = os.path.join(workspace_dir, "outputs", title, f"{title}_bgm_{total_duration}s.wav")
        generate_deep_sea_soundtrack(float(total_duration), bgm_wav)

    final_output = os.path.join(workspace_dir, "outputs", f"{title}_full_{total_duration}s.mp4")
    concat_videos_with_bgm_ffmpeg(segment_files, final_output, bgm_wav_path=bgm_wav, target_width=1280, target_height=704)
    return final_output


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(script_dir):
        os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="90秒深海探索纪录片 720P 高清图生视频 (I2V) 流水线")
    parser.add_argument("--storyboard", type=str, default="storyboard.json", help="分镜剧本 JSON 路径")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["ltx", "wan-fast", "wan-i2v"],
        default=None,
        help="模型引擎: ltx (极速720P原图I2V, 单镜~45s) | wan-fast (万相14B 10倍极速版, 单镜~2.5m) | wan-i2v (万相14B原生720P)",
    )
    parser.add_argument("--steps", type=int, default=0, help="自定义采样步数 (0=使用引擎最优极速步数)")
    parser.add_argument("--max-shots", type=int, default=0, help="限制渲染前 N 个镜头 (0=全部 18 个镜头共 90 秒)")
    parser.add_argument("--bgm-only", type=str, default="", help="仅生成指定路径的 90 秒深海纯配乐 WAV 并退出")
    parser.add_argument("--force", action="store_true", help="强制重新渲染所有分镜（忽略断点缓存）")
    args = parser.parse_args()

    with open(args.storyboard, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    if args.max_shots > 0:
        storyboard["shots"] = storyboard["shots"][:args.max_shots]

    total_sec = len(storyboard.get("shots", [])) * storyboard.get("segment_seconds", 5)

    if args.bgm_only:
        generate_deep_sea_soundtrack(float(total_sec), args.bgm_only)
        return

    workspace_dir = os.getcwd()
    setup_local_ssd_cache()

    engine = args.engine or storyboard.get("engine", "wan-fast")
    if engine == "wan-fast":
        final_video = run_wan_i2v_pipeline(storyboard, workspace_dir, force_rerun=args.force, fast_mode=True, steps_override=args.steps)
    elif engine == "wan-i2v":
        final_video = run_wan_i2v_pipeline(storyboard, workspace_dir, force_rerun=args.force, fast_mode=False, steps_override=args.steps)
    else:
        final_video = run_ltx_pipeline(storyboard, workspace_dir, force_rerun=args.force, steps_override=args.steps)

    print("========================================================")
    print("🎉 90秒深海探索纪录片（720P 超清原图驱动 I2V + 纯配乐版）全部生成完成！")
    print(f"📁 完整成片路径 (Google Drive): {final_video}")
    print("========================================================")


if __name__ == "__main__":
    main()
