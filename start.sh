#!/bin/zsh
# 死活题批改 App 启动脚本
cd "$(dirname "$0")"
exec /Users/yangyang/.workbuddy/binaries/python/envs/default/bin/python web/app.py
