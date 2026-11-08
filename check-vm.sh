#!/usr/bin/env bash
set -e

echo "========================================================"
echo "🚀 Google Antigravity 2.29 Remote VM 硬件与环境体检报告"
echo "========================================================"
echo ""

echo "🔹 [1] 系统与内核信息 (OS & Kernel):"
uname -a
cat /etc/os-release 2>/dev/null | grep -E "^(PRETTY_NAME|VERSION)=" || true
echo ""

echo "🔹 [2] CPU 处理器信息 (Xeon 型号与核心数):"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E "Model name|Socket|Thread|NUMA|CPU\(s\)|Vendor ID|Architecture|MHz"
else
    grep -E "model name|cpu cores|siblings" /proc/cpuinfo | head -n 6
fi
echo ""

echo "🔹 [3] 内存与交换空间 (RAM & Swap):"
free -h 2>/dev/null || vm_stat
echo ""

echo "🔹 [4] 磁盘容量与挂载点 (Disk Storage):"
df -h / /home/antigravity /shared 2>/dev/null | awk 'NR==1 || !seen[$0]++' || df -h /
echo ""

echo "🔹 [5] Google Drive 云盘挂载与配额 (Cloud Drive):"
if command -v gdrive >/dev/null 2>&1; then
    GDRIVE_BIN_DIR="${ANTIGRAVITY_APP_DATA_DIR:-$HOME/.gemini/jetski}/plugin_data/gdrive/bin"
    if ! ls "$GDRIVE_BIN_DIR"/gdrive-* >/dev/null 2>&1; then
        gdrive --version >/dev/null 2>&1 || true
        TMP_BIN=$(ls -t "$GDRIVE_BIN_DIR"/.gdrive.tmp.* 2>/dev/null | head -n 1 || true)
        CLI_VER=$(grep -E "^CLI_VERSION=" /opt/antigravity/bin/gdrive 2>/dev/null | cut -d= -f2 || echo "1013145354")
        if [ -n "$TMP_BIN" ] && [ -f "$TMP_BIN" ]; then
            mv -f "$TMP_BIN" "$GDRIVE_BIN_DIR/gdrive-$CLI_VER" 2>/dev/null || true
            rm -f "$GDRIVE_BIN_DIR"/.gdrive.tmp.* 2>/dev/null || true
        fi
    fi
    if gdrive auth status >/dev/null 2>&1; then
        echo "  - 状态 : 已连接 (Logged in)"
        gdrive readonly quota 2>/dev/null | sed 's/^/  - /' || true
    else
        echo "  - 状态 : 未登录或授权已过期"
    fi
else
    echo "  - 状态 : 未检测到 gdrive CLI"
fi
echo ""

echo "🔹 [6] 预装开发工具链版本 (Toolchain):"
echo "  - Node.js : $(node -v 2>/dev/null || echo '未安装')"
echo "  - Python3 : $(python3 --version 2>/dev/null || echo '未安装')"
echo "  - Git     : $(git --version 2>/dev/null || echo '未安装')"
echo "  - Docker  : $(docker --version 2>/dev/null || echo '未安装')"
echo "  - Go      : $(go version 2>/dev/null || echo '未安装')"
echo "  - Rust    : $(rustc --version 2>/dev/null || echo '未安装')"
echo ""

echo "========================================================"
echo "✅ 体检完成！当前工作目录: $(pwd)"
echo "========================================================"
