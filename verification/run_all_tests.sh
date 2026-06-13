#!/usr/bin/env bash
# Run pytest for all visualization verification subfolders.
# Usage: bash verification/run_all_tests.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"
echo "=== Running all visualization verification tests ==="
"$VENV" -m pytest "$SCRIPT_DIR" -v --tb=short
echo "=== All tests passed ==="
