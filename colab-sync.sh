#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Antigravity Cloud VM <-> Google Drive (20TB) <-> Google Colab GPU 协同工具
# ============================================================================

WORKSPACE_NAME="Antigravity_Workspace"
COLAB_FOLDER_NAME="Colab Notebooks"
NOTEBOOK_LOCAL="colab/Antigravity_GPU_Worker.ipynb"
TASK_SCRIPT_LOCAL="colab/gpu_task.py"
VIDEO_PIPELINE_LOCAL="colab/long_video_pipeline.py"
STORYBOARD_LOCAL="colab/storyboard.json"
STATE_FILE=".colab_drive_state.env"

ensure_gdrive() {
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
}

parse_first_id() {
    python3 -c '
import sys, json
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)
try:
    data = json.loads(raw)
    if isinstance(data, list) and data:
        print(data[0].get("id", ""))
    elif isinstance(data, dict):
        if "id" in data:
            print(data["id"])
        elif "files" in data and data["files"]:
            print(data["files"][0].get("id", ""))
except Exception:
    pass
'
}

get_or_create_folder() {
    local folder_name="$1"
    local parent_id="${2:-}"
    local folder_id=""

    if [ -n "$parent_id" ]; then
        folder_id=$(gdrive readonly search --name-exact "$folder_name" --parent-id "$parent_id" --type folder --max 5 --json 2>/dev/null | parse_first_id || true)
    else
        folder_id=$(gdrive readonly search --name-exact "$folder_name" --type folder --max 5 --json 2>/dev/null | parse_first_id || true)
    fi

    if [ -z "$folder_id" ]; then
        if [ -n "$parent_id" ]; then
            gdrive mutate mkdir "$folder_name" --parent "$parent_id" >/dev/null
            folder_id=$(gdrive readonly search --name-exact "$folder_name" --parent-id "$parent_id" --type folder --max 5 --json 2>/dev/null | parse_first_id || true)
        else
            gdrive mutate mkdir "$folder_name" >/dev/null
            folder_id=$(gdrive readonly search --name-exact "$folder_name" --type folder --max 5 --json 2>/dev/null | parse_first_id || true)
        fi
    fi
    echo "$folder_id"
}

upload_or_update_file() {
    local local_path="$1"
    local parent_id="$2"
    local mime_type="${3:-}"
    local filename
    filename=$(basename "$local_path")

    local existing_id=""
    existing_id=$(gdrive readonly search --name-contains "${filename%.*}" --parent-id "$parent_id" --max 5 --json 2>/dev/null | parse_first_id || true)

    if [ -n "$existing_id" ]; then
        if [ -n "$mime_type" ]; then
            gdrive mutate update "$existing_id" "$local_path" --mime-type "$mime_type" >/dev/null
        else
            gdrive mutate update "$existing_id" "$local_path" >/dev/null
        fi
        echo "$existing_id"
    else
        if [ -n "$mime_type" ]; then
            gdrive mutate upload "$local_path" --parent "$parent_id" --mime-type "$mime_type" >/dev/null
        else
            gdrive mutate upload "$local_path" --parent "$parent_id" >/dev/null
        fi
        existing_id=$(gdrive readonly search --name-contains "${filename%.*}" --parent-id "$parent_id" --max 5 --json 2>/dev/null | parse_first_id || true)
        echo "$existing_id"
    fi
}

cmd_init() {
    ensure_gdrive
    echo "========================================================"
    echo "🚀 正在初始化 Google Drive (20TB) + Colab GPU 长视频工作站..."
    echo "========================================================"

    COLAB_DIR_ID=$(get_or_create_folder "$COLAB_FOLDER_NAME")
    echo "✅ Colab 目录 [$COLAB_FOLDER_NAME] ID: $COLAB_DIR_ID"

    WORKSPACE_DIR_ID=$(get_or_create_folder "$WORKSPACE_NAME")
    echo "✅ 云端共享工作区 [$WORKSPACE_NAME] ID: $WORKSPACE_DIR_ID"

    OUTPUTS_DIR_ID=$(get_or_create_folder "outputs" "$WORKSPACE_DIR_ID")
    echo "✅ 云端输出目录 [$WORKSPACE_NAME/outputs] ID: $OUTPUTS_DIR_ID"

    NOTEBOOK_ID=$(upload_or_update_file "$NOTEBOOK_LOCAL" "$COLAB_DIR_ID" "application/vnd.google.colaboratory")
    echo "✅ Colab GPU 笔记本已更新 ID: $NOTEBOOK_ID"

    TASK_ID=$(upload_or_update_file "$TASK_SCRIPT_LOCAL" "$WORKSPACE_DIR_ID")
    echo "✅ GPU 基准脚本 [gpu_task.py] 已同步 ID: $TASK_ID"

    VIDEO_PIPELINE_ID=$(upload_or_update_file "$VIDEO_PIPELINE_LOCAL" "$WORKSPACE_DIR_ID")
    echo "✅ 1分钟+ 长视频流水线 [long_video_pipeline.py] 已同步 ID: $VIDEO_PIPELINE_ID"

    STORYBOARD_ID=$(upload_or_update_file "$STORYBOARD_LOCAL" "$WORKSPACE_DIR_ID")
    echo "✅ 60秒分镜剧本 [storyboard.json] 已同步 ID: $STORYBOARD_ID"

    cat > "$STATE_FILE" <<EOF
COLAB_DIR_ID=$COLAB_DIR_ID
WORKSPACE_DIR_ID=$WORKSPACE_DIR_ID
OUTPUTS_DIR_ID=$OUTPUTS_DIR_ID
NOTEBOOK_ID=$NOTEBOOK_ID
TASK_ID=$TASK_ID
VIDEO_PIPELINE_ID=$VIDEO_PIPELINE_ID
STORYBOARD_ID=$STORYBOARD_ID
EOF

    echo ""
    echo "🎉 长视频工作站部署完成！"
    echo "========================================================"
}

cmd_status() {
    ensure_gdrive
    echo "========================================================"
    echo "📊 Google Drive & Colab GPU 协同状态"
    echo "========================================================"
    gdrive auth status
    echo ""
    echo "🔹 云盘容量配额 (Quota):"
    gdrive readonly quota
    echo ""
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
        echo "🔹 Colab GPU Notebook ID: $NOTEBOOK_ID"
        echo ""
        echo "🔹 云端工作区 [$WORKSPACE_NAME] 文件列表:"
        gdrive readonly ls "$WORKSPACE_DIR_ID" --max 20
        echo ""
        echo "🔹 云端输出目录 [$WORKSPACE_NAME/outputs] 文件列表:"
        gdrive readonly ls "$OUTPUTS_DIR_ID" --max 20
    else
        echo "⚠️ 尚未初始化，请先运行: bash ./colab-sync.sh init"
    fi
}

cmd_push() {
    ensure_gdrive
    local target="${1:-}"
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "❌ 用法: bash ./colab-sync.sh push <本地文件路径>"
        exit 1
    fi
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
    else
        WORKSPACE_DIR_ID=$(get_or_create_folder "$WORKSPACE_NAME")
    fi
    local file_id
    if [[ "$target" == *.ipynb ]]; then
        COLAB_DIR_ID=$(get_or_create_folder "$COLAB_FOLDER_NAME")
        file_id=$(upload_or_update_file "$target" "$COLAB_DIR_ID" "application/vnd.google.colaboratory")
        echo "✅ Notebook 已推送至 Colab Notebooks! (ID: $file_id)"
    else
        file_id=$(upload_or_update_file "$target" "$WORKSPACE_DIR_ID")
        echo "✅ 文件 [$target] 已同步至 Google Drive [$WORKSPACE_NAME] (ID: $file_id)"
    fi
}

cmd_pull() {
    ensure_gdrive
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
    else
        WORKSPACE_DIR_ID=$(get_or_create_folder "$WORKSPACE_NAME")
        OUTPUTS_DIR_ID=$(get_or_create_folder "outputs" "$WORKSPACE_DIR_ID")
    fi
    mkdir -p ./colab/outputs
    echo "📥 正在从 Google Drive [$WORKSPACE_NAME/outputs] 拉取生成的视频与结果..."
    local files_json
    files_json=$(gdrive readonly ls "$OUTPUTS_DIR_ID" --max 50 --json 2>/dev/null || echo "[]")
    python3 - "$files_json" <<'PYEOF'
import json, subprocess, sys
raw = sys.argv[1].strip()
data = json.loads(raw) if raw else []
files = data if isinstance(data, list) else (data.get("files") or [])
if not files:
    print("  (当前 outputs 目录为空，请先在 Colab GPU 中运行 Notebook)")
    sys.exit(0)
for f in files:
    if f.get("mimeType") == "application/vnd.google-apps.folder":
        continue
    fid = f["id"]
    name = f["name"]
    out = f"./colab/outputs/{name}"
    print(f"  -> 下载 {name} ({fid}) 至 {out}")
    subprocess.run(["gdrive", "readonly", "download", fid, "--out", out], check=True)
PYEOF
    echo "✅ 拉取完成！本地目录: ./colab/outputs/"
    ls -la ./colab/outputs/
}

case "${1:-status}" in
    init)   cmd_init ;;
    status) cmd_status ;;
    push)   shift; cmd_push "${1:-}" ;;
    pull)   cmd_pull ;;
    *)
        echo "用法: bash ./colab-sync.sh {init|status|push <file>|pull}"
        exit 1
        ;;
esac
