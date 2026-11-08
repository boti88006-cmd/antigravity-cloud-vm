#!/usr/bin/env python3
"""
《深海奥德赛 Deep Sea Odyssey》90秒深海探索纪录片 (纯配乐无配音版)
高速水下体积光、生物荧光粒子 (3D Bioluminescent Plankton) 与 18 镜头连续运镜合成器
"""
import glob
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image


def load_keyframes(artifact_dir: str, width: int = 1280, height: int = 720) -> list[np.ndarray]:
    """加载按深海下潜顺序排列的 9 张 IMAX 16:9 关键帧图像，并缩放为超采样尺寸供 18 个连续镜头推拉摇移"""
    prefixes = [
        "deep_sea_act1_sunlight",
        "deep_sea_act1_submersible",
        "deep_sea_act2_jellyfish",
        "deep_sea_comb_jelly",
        "deep_sea_act2_midnight",
        "deep_sea_act3_vents",
        "deep_sea_trench_garden",
        "deep_sea_act3_hadal_arch",
        "deep_sea_finale_galaxy",
    ]
    # 超采样尺寸 (1.35x) 以便平滑运镜 (Zoom / Pan / Tilt)
    overscan_w = int(width * 1.35)
    overscan_h = int(height * 1.35)

    frames = []
    for p in prefixes:
        matches = sorted(glob.glob(os.path.join(artifact_dir, f"{p}_*.jpg")))
        if not matches:
            raise FileNotFoundError(f"Missing keyframe for prefix: {p}")
        img = Image.open(matches[-1]).convert("RGB").resize((overscan_w, overscan_h), Image.Resampling.LANCZOS)
        frames.append(np.asarray(img, dtype=np.float32))
    return frames


def main():
    artifact_dir = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/06d05825-8980-46fd-ac20-6235d09f9dce"
    output_dir = "/home/antigravity/boti88006-cmd/antigravity-cloud-vm/colab/outputs"
    segments_dir = os.path.join(output_dir, "Deep_Sea_Odyssey_90s", "segments")
    os.makedirs(segments_dir, exist_ok=True)

    bgm_wav = os.path.join(output_dir, "Deep_Sea_Odyssey_90s", "Deep_Sea_Odyssey_90s_bgm_90s.wav")
    final_mp4 = os.path.join(output_dir, "Deep_Sea_Odyssey_90s_full_90s.mp4")

    width, height = 1280, 720
    fps = 24
    total_seconds = 90
    total_frames = total_seconds * fps  # 2160 帧 (18 个 5 秒连续镜头)

    print("🖼️ 正在加载 9 幅 IMAX 深海实景母版图层并构建 18 个连续 5 秒分镜运镜轨迹...")
    keyframes = load_keyframes(artifact_dir, width=width, height=height)
    overscan_h, overscan_w, _ = keyframes[0].shape

    # 为 9 张母版各分配 2 个连续运镜分镜 (9 x 2 = 18 个 5 秒分镜 = 90 秒)
    # 运镜模式: (start_zoom, end_zoom, start_rx, end_rx, start_ry, end_ry)
    # rx, ry in [0.0, 1.0] 表示裁剪窗口中心在超采样画布上的相对偏移
    shot_motions = [
        # Scene 1: Sunlight Zone (Shot 1 & Shot 2)
        (0, 1.00, 0.88, 0.20, 0.45, 0.15, 0.45),
        (0, 0.88, 0.78, 0.45, 0.85, 0.45, 0.80),
        # Scene 2: Submersible Descent (Shot 3 & Shot 4)
        (1, 0.96, 0.86, 0.15, 0.50, 0.15, 0.55),
        (1, 0.86, 0.76, 0.50, 0.82, 0.55, 0.88),
        # Scene 3: Bioluminescent Jellyfish (Shot 5 & Shot 6)
        (2, 0.98, 0.86, 0.50, 0.50, 0.85, 0.45),
        (2, 0.86, 0.75, 0.50, 0.30, 0.45, 0.15),
        # Scene 4: Comb Jelly Macro (Shot 7 & Shot 8)
        (3, 0.95, 0.82, 0.20, 0.55, 0.50, 0.45),
        (3, 0.82, 0.92, 0.55, 0.85, 0.45, 0.60),
        # Scene 5: Giant Squid & Anglerfish Midnight Zone (Shot 9 & Shot 10)
        (4, 0.96, 0.85, 0.12, 0.52, 0.25, 0.45),
        (4, 0.85, 0.76, 0.52, 0.88, 0.45, 0.82),
        # Scene 6: Hydrothermal Black Smoker Vents (Shot 11 & Shot 12)
        (5, 0.96, 0.85, 0.50, 0.48, 0.88, 0.48),
        (5, 0.85, 0.75, 0.48, 0.52, 0.48, 0.12),
        # Scene 7: Mariana Trench Glass Sponge Garden & Snailfish (Shot 13 & Shot 14)
        (6, 0.95, 0.84, 0.18, 0.55, 0.60, 0.50),
        (6, 0.84, 0.76, 0.55, 0.88, 0.50, 0.42),
        # Scene 8: Hadal Basalt Arch & Bioluminescent Vortex (Shot 15 & Shot 16)
        (7, 0.98, 0.86, 0.50, 0.50, 0.75, 0.48),
        (7, 0.86, 0.74, 0.50, 0.52, 0.48, 0.22),
        # Scene 9: Finale Underwater Bioluminescent Galaxy (Shot 17 & Shot 18)
        (8, 0.78, 0.88, 0.50, 0.50, 0.75, 0.45),
        (8, 0.88, 1.00, 0.50, 0.50, 0.45, 0.15),
    ]

    # 初始化 3D 深海生物荧光浮游粒子 (Marine Snow & Bioluminescence)
    rng = np.random.default_rng(seed=88006)
    num_particles = 140
    px = rng.uniform(0, width, num_particles).astype(np.float32)
    py = rng.uniform(0, height, num_particles).astype(np.float32)
    p_speed_y = rng.uniform(-45.0, -18.0, num_particles).astype(np.float32)  # 向上漂浮 (配合下潜感)
    p_speed_x = rng.uniform(-8.0, 8.0, num_particles).astype(np.float32)
    p_phase = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)
    p_radius = rng.integers(1, 4, num_particles)
    # 荧光色彩 (青蓝、冰蓝、洋红微光)
    palette = np.array([
        [110.0, 235.0, 255.0],
        [160.0, 245.0, 255.0],
        [90.0,  190.0, 255.0],
        [210.0, 150.0, 255.0],
    ], dtype=np.float32)
    p_colors = palette[rng.integers(0, len(palette), num_particles)]

    # 预计算水下焦散/光束横向坐标网格
    x_grid = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y_grid = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    # 启动 FFmpeg 管道直接编码 90 秒 720p H.264 视频并与 90s 立体声深海配乐混流
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-i", bgm_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "19",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_mp4,
    ]

    print(f"🎬 开始渲染 90 秒 (共 {total_frames} 帧, 1280x720 @ {fps}fps) 深海探索纪录片并混流立体声配乐...")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames_per_shot = 5 * fps  # 120 帧 = 5 秒
    crossfade_frames = 18      # 场景切换处 0.75 秒柔滑交叉溶解

    def sample_shot_frame(shot_idx: int, local_progress: float) -> np.ndarray:
        kf_idx, z0, z1, rx0, rx1, ry0, ry1 = shot_motions[shot_idx]
        # 余弦平滑插值 (Ease-in-out)
        s = 0.5 - 0.5 * np.cos(np.pi * np.clip(local_progress, 0.0, 1.0))
        zoom = z0 + (z1 - z0) * s
        rx = rx0 + (rx1 - rx0) * s
        ry = ry0 + (ry1 - ry0) * s

        crop_w = int(width * zoom)
        crop_h = int(height * zoom)
        max_x = max(overscan_w - crop_w, 1)
        max_y = max(overscan_h - crop_h, 1)
        x0 = int(np.clip(rx * max_x, 0, max_x))
        y0 = int(np.clip(ry * max_y, 0, max_y))

        sub_img = keyframes[kf_idx][y0:y0 + crop_h, x0:x0 + crop_w]
        pil_sub = Image.fromarray(sub_img.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(pil_sub, dtype=np.float32)

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        shot_idx = min(f_idx // frames_per_shot, len(shot_motions) - 1)
        in_shot_frame = f_idx % frames_per_shot
        progress = in_shot_frame / float(frames_per_shot)

        frame = sample_shot_frame(shot_idx, progress)

        # 在不同母版切换的分镜边界添加平滑交叉溶解转场
        if shot_idx < len(shot_motions) - 1:
            curr_kf = shot_motions[shot_idx][0]
            next_kf = shot_motions[shot_idx + 1][0]
            if curr_kf != next_kf and in_shot_frame >= (frames_per_shot - crossfade_frames):
                alpha = (in_shot_frame - (frames_per_shot - crossfade_frames)) / float(crossfade_frames)
                next_frame = sample_shot_frame(shot_idx + 1, 0.0)
                frame = (1.0 - alpha) * frame + alpha * next_frame

        # 叠加动态水下体积光律动 (Dynamic Volumetric Caustics / Deep Sea Pulse)
        caustic_wave = (
            0.5 + 0.5 * np.sin(2.0 * np.pi * (2.5 * x_grid - 0.35 * t_sec + 0.6 * y_grid))
        ) * (1.0 - 0.65 * y_grid)
        # 阳光带增强金青色丁达尔光束，深海带增强幽蓝生物光脉冲
        if t_sec < 20.0:
            light_boost = caustic_wave[:, :, None] * np.array([12.0, 22.0, 26.0], dtype=np.float32)
        else:
            pulse = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.4 * t_sec)
            light_boost = (caustic_wave[:, :, None] * pulse) * np.array([4.0, 16.0, 28.0], dtype=np.float32)
        frame = frame + light_boost

        # 叠加 3D 景深深海雪与生物荧光浮游生物粒子
        cur_px = (px + p_speed_x * t_sec + 12.0 * np.sin(t_sec * 0.9 + p_phase)) % width
        cur_py = (py + p_speed_y * t_sec) % height
        twinkle = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.7 * t_sec + p_phase))

        ix = cur_px.astype(np.int32)
        iy = cur_py.astype(np.int32)
        for p_i in range(num_particles):
            r = p_radius[p_i]
            x_c, y_c = ix[p_i], iy[p_i]
            x1, x2 = max(0, x_c - r), min(width, x_c + r + 1)
            y1, y2 = max(0, y_c - r), min(height, y_c + r + 1)
            color = p_colors[p_i] * twinkle[p_i]
            frame[y1:y2, x1:x2] = np.minimum(255.0, frame[y1:y2, x1:x2] + color * 0.75)

        # 首尾 2.5 秒黑场淡入淡出
        if t_sec < 2.5:
            frame *= (t_sec / 2.5)
        elif t_sec > (total_seconds - 3.0):
            frame *= max(0.0, (total_seconds - t_sec) / 3.0)

        out_bytes = np.clip(frame, 0.0, 255.0).astype(np.uint8).tobytes()
        proc.stdin.write(out_bytes)

        if (f_idx + 1) % (fps * 15) == 0:
            print(f"  -> 已渲染 {int((f_idx + 1) / fps)} / {total_seconds} 秒 ({(f_idx + 1) * 100 // total_frames}%)")

    proc.stdin.close()
    proc.wait()
    elapsed = time.time() - t0
    print(f"✅ 90 秒深海探索纪录片渲染与配乐混流完成！耗时: {elapsed:.1f}s")
    print(f"📁 成片路径: {final_mp4} (大小: {os.path.getsize(final_mp4) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
