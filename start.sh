#!/bin/zsh
# 死活题批改 App 启动脚本
# 指定 Python 解释器：export PYTHON_BIN=/path/to/python（默认 python3）
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-python3}" web/app.py
