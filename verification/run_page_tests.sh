#!/usr/bin/env bash
# Browser-level smoke tests for all topic HTML pages.
# Uses Playwright (headless Chromium) via the project virtualenv.
#
# Run from project root:
#   bash verification/run_page_tests.sh
#
# First-time setup (one-off):
#   verification/.venv/bin/pip install playwright
#   verification/.venv/bin/playwright install chromium

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
PYTEST="$SCRIPT_DIR/.venv/bin/pytest"

cd "$PROJECT_ROOT"

echo "=== Browser smoke tests (Playwright / headless Chromium) ==="
"$PYTEST" verification/page_tests/ -v --tb=short "$@"
