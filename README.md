# Antigravity 2.29 Cloud VM Workspace (`antigravity-cloud-vm`)

专用于 **Google Antigravity 2.29 Remote Agents（云端远程智能体 / 独立云端虚拟机）** 的挂载工作区仓库，已集成 **20TB Google Drive 云盘** 与 **Google Colab GPU** 算力协同桥接工具。

## 1. 快速体检（查看 CPU / 内存 / 多层磁盘 / Google Drive 配额）

在 Antigravity 桌面端或网页端（https://antigravity.google.com/remote）绑定本仓库后，直接在对话框发送：
```text
体检
```
或运行：
```bash
bash ./check-vm.sh
```

## 2. Google Drive (20TB) + Google Colab GPU 协同工作流

通过内置的 `colab-sync.sh` 脚本，可将当前 VM 编写的代码与 Notebook 直接投递至您的 **Google Drive (`Colab Notebooks` & `Antigravity_Workspace`)**，由 **Google Colab GPU (T4 / L4 / A100)** 挂载执行并回传结果：

```bash
# 1. 初始化云端工作区并部署 Colab GPU 笔记本（输出一键打开的 Colab 链接）
bash ./colab-sync.sh init

# 2. 查看 Google Drive 20TB 配额与云端工作区文件列表
bash ./colab-sync.sh status

# 3. 推送本地脚本、数据集或新 Notebook 到 Google Drive / Colab
bash ./colab-sync.sh push ./colab/gpu_task.py

# 4. 从 Google Drive (Antigravity_Workspace/outputs) 拉取 Colab GPU 计算结果到本地
bash ./colab-sync.sh pull
```
