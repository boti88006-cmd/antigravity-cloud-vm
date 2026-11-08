#!/usr/bin/env python3
"""
Antigravity Cloud VM -> Google Colab GPU 算力任务脚本
在 Colab GPU 中挂载 Google Drive 后运行，并将 GPU 算力基准与执行结果写回 outputs/gpu_report.json。
"""
import json
import os
import platform
import time
from datetime import datetime, timezone

def main():
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "fp16_tflops": None,
    }

    try:
        import torch
        report["pytorch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)

            # 执行 FP16 矩阵乘法算力测试 (4096 x 4096)
            n = 4096
            a = torch.randn((n, n), device=device, dtype=torch.float16)
            b = torch.randn((n, n), device=device, dtype=torch.float16)
            # 预热
            for _ in range(5):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()

            iters = 20
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            flops = 2 * (n ** 3) * iters
            tflops = (flops / elapsed) / 1e12
            report["fp16_tflops"] = round(tflops, 2)
    except Exception as e:
        report["error"] = str(e)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "gpu_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("========================================================")
    print("⚡ Google Colab GPU 任务执行完成！")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"📄 结果已保存至 Google Drive: {os.path.abspath(out_path)}")
    print("👉 现在可回到 Antigravity VM 运行: bash ./colab-sync.sh pull")
    print("========================================================")

if __name__ == "__main__":
    main()
