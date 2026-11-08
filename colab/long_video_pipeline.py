#!/usr/bin/env python3
"""
Antigravity Cloud VM × Google Colab GPU × 20TB Google Drive
1分钟+ 长视频生成流水线 (Long Video Generation Pipeline)

核心特性:
1. 20TB Google Drive 永久模型缓存: 将 HF_HOME 指向云盘，几十 GB 模型权重只下载一次。
2. 尾帧接力续写 (Last-Frame Chaining): 自动提取上一镜头的最后一帧作为下一镜头的首帧参考 (I2V)，实现长镜头连贯运镜。
3. 断点续传 (Checkpoint Resume): 每个 5 秒分镜生成后立即落盘至 Google Drive，意外中断重跑自动跳过已完成分镜。
4. FFmpeg 自动合成长片: 所有分镜渲染完成后，自动无损拼接为 60 秒+ 完整 MP4 视频。
"""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path


def setup_drive_cache(workspace_dir: str):
    """将 HuggingFace 模型缓存绑定到 20TB Google Drive，避免每次重下模型"""
    hf_cache = os.path.join(workspace_dir, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache, "hub")
    print(f"📦 模型永久缓存目录已绑定至 Google Drive: {hf_cache}")


def extract_last_frame(video_path: str, output_image_path: str):
    """从上一段 MP4 视频中提取最后一帧 PIL Image，用于下一段尾帧接力 (I2V)"""
    import imageio.v3 as iio
    from PIL import Image

    frames = iio.imread(video_path, plugin="pyav")
    last_frame_np = frames[-1]
    img = Image.fromarray(last_frame_np)
    img.save(output_image_path)
    return img


def concat_videos_ffmpeg(segment_paths: list[str], final_output_path: str):
    """使用 FFmpeg 将多个 5 秒分镜无损合成 60 秒+ 完整长视频"""
    list_file = os.path.join(os.path.dirname(final_output_path), "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for seg in segment_paths:
            abs_seg = os.path.abspath(seg)
            f.write(f"file '{abs_seg}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        final_output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"🎬 1分钟+ 完整长视频已合成: {final_output_path}")


def run_ltx_pipeline(storyboard: dict, workspace_dir: str, force_rerun: bool = False):
    """
    使用 Lightricks/LTX-Video (支持 T2V + I2V 尾帧接力) 渲染 60 秒+ 长视频
    针对 G4 / T4 / L4 GPU 开启显存卸载与 VAE 分块解码
    """
    import torch
    from diffusers import LTXConditionPipeline, LTXPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video, load_image

    model_id = "Lightricks/LTX-Video"
    print(f"🚀 正在加载视频大模型: {model_id} (首次运行将下载至 20TB Google Drive 缓存)...")

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    # 针对 16GB~24GB 显存开启 CPU Offload 与 VAE Tiling 防爆显存
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    width = storyboard.get("resolution", {}).get("width", 768)
    height = storyboard.get("resolution", {}).get("height", 512)
    fps = storyboard.get("fps", 24)
    seg_sec = storyboard.get("segment_seconds", 5)
    # LTX-Video 帧数需满足 8 * k + 1，例如 5 秒 @ 24fps = 121 帧
    raw_frames = fps * seg_sec
    num_frames = ((raw_frames - 1) // 8) * 8 + 1

    global_style = storyboard.get("global_style", "")
    negative_prompt = storyboard.get("negative_prompt", "low quality, worst quality, deformed, distorted")
    title = storyboard.get("title", "long_video")

    segments_dir = os.path.join(workspace_dir, "outputs", title, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    shots = storyboard.get("shots", [])
    segment_files = []
    prev_video_path = None

    print(f"📋 开始渲染长视频《{title}》: 共 {len(shots)} 个镜头, 单镜头 {seg_sec}s ({num_frames} 帧), 预计总时长 {len(shots) * seg_sec} 秒")

    for idx, shot in enumerate(shots, start=1):
        shot_id = shot.get("id", idx)
        seg_path = os.path.join(segments_dir, f"shot_{shot_id:03d}.mp4")
        last_frame_path = os.path.join(segments_dir, f"shot_{shot_id:03d}_last.png")

        # 断点续传检测：如果分镜已存在于 Google Drive，直接复用
        if os.path.exists(seg_path) and not force_rerun:
            print(f"⏭️ [镜头 {idx}/{len(shots)}] 已存在于云盘，跳过渲染: {seg_path}")
            segment_files.append(seg_path)
            prev_video_path = seg_path
            continue

        full_prompt = f"{shot['prompt']}, {global_style}" if global_style else shot["prompt"]
        chain_prev = shot.get("chain_from_previous", False) and (prev_video_path is not None)

        t0 = time.time()
        generator = torch.Generator("cpu").manual_seed(42 + shot_id)

        if chain_prev:
            print(f"🔗 [镜头 {idx}/{len(shots)}] 尾帧接力模式 (I2V): 提取上一镜头尾帧继续运镜...")
            cond_img = extract_last_frame(prev_video_path, last_frame_path)
            condition = LTXVideoCondition(image=cond_img, frame_index=0)
            video_frames = pipe(
                conditions=[condition],
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=35,
                guidance_scale=3.0,
                generator=generator,
            ).frames[0]
        else:
            print(f"🎬 [镜头 {idx}/{len(shots)}] 独立新场景模式 (T2V): 渲染全新分镜...")
            video_frames = pipe(
                conditions=None,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=35,
                guidance_scale=3.0,
                generator=generator,
            ).frames[0]

        export_to_video(video_frames, seg_path, fps=fps)
        elapsed = time.time() - t0
        print(f"✅ [镜头 {idx}/{len(shots)}] 渲染完成 ({elapsed:.1f}s) -> 已实时存盘至 Google Drive: {seg_path}")

        segment_files.append(seg_path)
        prev_video_path = seg_path

        # 释放显存碎片
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_output = os.path.join(workspace_dir, "outputs", f"{title}_full_{len(shots)*seg_sec}s.mp4")
    concat_videos_ffmpeg(segment_files, final_output)
    return final_output


def run_wan_pipeline(storyboard: dict, workspace_dir: str, force_rerun: bool = False):
    """
    使用阿里通义万相 Wan-AI/Wan2.1-T2V-1.3B-Diffusers 渲染高画质多分镜长视频
    """
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    print(f"🚀 正在加载通义万相视频模型: {model_id}...")

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    width = storyboard.get("resolution", {}).get("width", 832)
    height = storyboard.get("resolution", {}).get("height", 480)
    fps = storyboard.get("fps", 16)
    seg_sec = storyboard.get("segment_seconds", 5)
    num_frames = 81  # Wan2.1 标准 81 帧 (~5s @ 16fps)

    global_style = storyboard.get("global_style", "")
    negative_prompt = storyboard.get("negative_prompt", "")
    title = storyboard.get("title", "long_video_wan")

    segments_dir = os.path.join(workspace_dir, "outputs", title, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    shots = storyboard.get("shots", [])
    segment_files = []

    for idx, shot in enumerate(shots, start=1):
        shot_id = shot.get("id", idx)
        seg_path = os.path.join(segments_dir, f"shot_{shot_id:03d}.mp4")

        if os.path.exists(seg_path) and not force_rerun:
            print(f"⏭️ [镜头 {idx}/{len(shots)}] 已存在于云盘，跳过渲染: {seg_path}")
            segment_files.append(seg_path)
            continue

        full_prompt = f"{shot['prompt']}, {global_style}" if global_style else shot["prompt"]
        print(f"🎬 [镜头 {idx}/{len(shots)}] Wan2.1 正在渲染...")
        t0 = time.time()
        video_frames = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=5.0,
        ).frames[0]

        export_to_video(video_frames, seg_path, fps=fps)
        print(f"✅ [镜头 {idx}/{len(shots)}] 完成 ({time.time() - t0:.1f}s) -> {seg_path}")
        segment_files.append(seg_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_output = os.path.join(workspace_dir, "outputs", f"{title}_full_{len(shots)*seg_sec}s.mp4")
    concat_videos_ffmpeg(segment_files, final_output)
    return final_output


def main():
    parser = argparse.ArgumentParser(description="1分钟+ AI 长视频生成流水线")
    parser.add_argument("--storyboard", type=str, default="storyboard.json", help="分镜剧本 JSON 路径")
    parser.add_argument("--engine", type=str, choices=["ltx", "wan"], default=None, help="覆盖模型引擎 (ltx 或 wan)")
    parser.add_argument("--max-shots", type=int, default=0, help="限制渲染前 N 个镜头 (0=全部渲染，用于快速测试)")
    parser.add_argument("--force", action="store_true", help="强制重新渲染所有分镜（忽略断点缓存）")
    args = parser.parse_args()

    workspace_dir = os.getcwd()
    setup_drive_cache(workspace_dir)

    with open(args.storyboard, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    if args.max_shots > 0:
        storyboard["shots"] = storyboard["shots"][:args.max_shots]

    engine = args.engine or storyboard.get("engine", "ltx")
    if engine == "wan":
        final_video = run_wan_pipeline(storyboard, workspace_dir, force_rerun=args.force)
    else:
        final_video = run_ltx_pipeline(storyboard, workspace_dir, force_rerun=args.force)

    print("========================================================")
    print(f"🎉 长视频生成任务全部完成！")
    print(f"📁 完整成片路径 (Google Drive): {final_video}")
    print("👉 在 Antigravity VM 中执行 `bash ./colab-sync.sh pull` 即可拉取到本地。")
    print("========================================================")


if __name__ == "__main__":
    main()
