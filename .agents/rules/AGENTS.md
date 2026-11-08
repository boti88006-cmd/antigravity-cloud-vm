---
trigger: always_on
---

# Antigravity Cloud VM (Remote Agent) 工作区规范

1. **语言偏好**：始终使用简体中文回复用户。
2. **快捷体检指令**：当用户发送“体检”、“查看配置”、“看看虚拟机配置”或“check”时，立即在终端执行 `bash ./check-vm.sh`，并将 CPU 型号（如 Intel Xeon 第四代）、内存大小（如 8GB RAM）、多层磁盘空间（`/home/antigravity` 19GB、`/shared` 1.0PB）、Google Drive (20TB) 配额及预装工具链汇总展示给用户。
3. **Colab GPU 与 Google Drive 协同**：当用户需要运行 GPU 任务或同步大文件至 Google Drive 时，使用 `bash ./colab-sync.sh {init|status|push|pull}` 将任务脚本或 `.ipynb` 笔记本同步至 Google Drive 的 `Antigravity_Workspace` 与 `Colab Notebooks`。
4. **代码与产物持久化**：由于云端虚拟机（Remote Agent VM）是按会话生命周期托管的隔离环境，当完成重要代码编写、数据抓取或构建产物后，主动提醒或协助用户通过 `git add`、`git commit` 和 `git push` 同步回 GitHub 仓库。
