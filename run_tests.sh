#!/usr/bin/env bash
# 运行全部复现 + 回归测试（仅标准库，Python 3.8+）
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest discover -s tests -v
