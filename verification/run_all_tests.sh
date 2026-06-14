#!/usr/bin/env bash
# Run all verification tests: math/structural (pytest) and browser smoke (Playwright).
# Usage: bash verification/run_all_tests.sh
#
# Pass --skip-browser to run only the math/structural tests (faster, no browser needed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTEST="$SCRIPT_DIR/.venv/bin/pytest"

SKIP_BROWSER=false
for arg in "$@"; do
  [[ "$arg" == "--skip-browser" ]] && SKIP_BROWSER=true
done

cd "$PROJECT_ROOT"

echo "=== Math & structural tests (pytest) ==="
"$PYTEST" "$SCRIPT_DIR" --ignore="$SCRIPT_DIR/page_tests" -v --tb=short

if [[ "$SKIP_BROWSER" == "true" ]]; then
  echo ""
  echo "=== Skipped browser smoke tests (--skip-browser) ==="
else
  echo ""
  echo "=== Browser smoke tests (Playwright / headless Chromium) ==="
  "$PYTEST" "$SCRIPT_DIR/page_tests" -v --tb=short
fi

echo ""
echo "=== All tests passed ==="
