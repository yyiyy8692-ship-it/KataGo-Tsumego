#!/bin/zsh
# 死活题 App 启动脚本（答案模式 + 批改模式共用一个 Flask 服务）
#
# 解释器选择顺序：
#   1) $PYTHON_BIN 显式指定
#   2) 能 import cv2 与 pymupdf 的解释器
#      —— 系统 python3 通常没装这两个，裸跑要等用户上传 PDF 时才 ImportError
# 用法：
#   ./start.sh                          # 自动挑解释器
#   PYTHON_BIN=/path/to/python ./start.sh
cd "$(dirname "$0")"

pick_python() {
  local candidates=(
    "${PYTHON_BIN:-}"
    "$HOME/.workbuddy/binaries/python/envs/default/bin/python"
    "$(command -v python3)"
  )
  local py
  for py in $candidates; do
    [ -n "$py" ] && [ -x "$py" ] || continue
    if "$py" -c "import cv2, pymupdf" >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  echo "python3"      # 兜底：跑不起来会在日志里看到明确 ImportError
}

PY="$(pick_python)"
echo "使用解释器：$PY"
echo "访问地址：http://$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1):${TSUME_PORT:-5050}"
exec "$PY" web/app.py
