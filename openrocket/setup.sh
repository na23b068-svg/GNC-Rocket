#!/usr/bin/env bash
# Fetch OpenRocket 23.09 (gitignored) and regenerate every rocket-designer output.
set -euo pipefail
cd "$(dirname "$0")"
JAR=bin/OpenRocket-23.09.jar
if [ ! -f "$JAR" ]; then
  mkdir -p bin
  curl -fL -o "$JAR.part" https://github.com/openrocket/openrocket/releases/download/release-23.09/OpenRocket-23.09.jar
  mv "$JAR.part" "$JAR"
fi
[ "${1:-}" = "--jar-only" ] && exit 0
../.venv/bin/python run_openrocket.py
