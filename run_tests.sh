#!/usr/bin/env bash
# Regression runner for the eye_search RK3576 NPU port.
#
#   ./run_tests.sh            # smoke: 20 video frames
#   ./run_tests.sh --full     # whole video (1615 frames)
#   ./run_tests.sh --quick    # model chain only (no video cases)
#
# Two layers:
#
#   L1  model chain - checks the ONNX -> .rknn conversion chain
#   L2  NPU layer   - checks the Rockchip NPU inference path
#
# A layer that cannot run on the current machine is reported as [SKIP]
# instead of a failure.  Every case writes its output to
# output/tests/<case>.log.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi

LOG_DIR="$ROOT/output/tests"
mkdir -p "$LOG_DIR"

FRAMES=20
RUN_QUICK=0
for arg in "$@"; do
    case "$arg" in
        --full) FRAMES=0 ;;
        --quick) RUN_QUICK=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
    esac
done

RKNN="$ROOT/assets/eyegaze_rk3576.rknn"
BACKBONE="$ROOT/assets/eyegaze_backbone.onnx"

PASS=0
FAIL=0
SKIP=0

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_BAD=$'\033[31m'; C_SKIP=$'\033[33m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_BAD=""; C_SKIP=""; C_OFF=""
fi

pass() { printf "  %s[PASS]%s %-42s %s\n" "$C_OK" "$C_OFF" "$1" "${2:-}"; PASS=$((PASS + 1)); }
fail() { printf "  %s[FAIL]%s %-42s %s\n" "$C_BAD" "$C_OFF" "$1" "${2:-}"; FAIL=$((FAIL + 1)); }
skip() { printf "  %s[SKIP]%s %-42s %s\n" "$C_SKIP" "$C_OFF" "$1" "${2:-}"; SKIP=$((SKIP + 1)); }
head_() { printf "\n== %s ==\n" "$1"; }

# run <case-name> <command...>  -> stores the log path in LAST_LOG
run() {
    local name="$1"; shift
    LAST_LOG="$LOG_DIR/${name}.log"
    "$@" > "$LAST_LOG" 2>&1
    LAST_RC=$?
    return $LAST_RC
}

# number <log> <line-regex> <field-regex>  -> first number of the first match
number() {
    grep -E "$2" "$1" 2>/dev/null | head -1 \
        | grep -oE "$3" 2>/dev/null | head -1 \
        | grep -oE "[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?" | head -1
}

# lt <value> <limit>  -> true when value < limit (empty value -> false)
lt() {
    [ -n "${1:-}" ] || return 1
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'
}

echo "eye_search test runner (RK3576 NPU)"
echo "  python : $PY"
echo "  frames : $([ "$FRAMES" -eq 0 ] && echo "all" || echo "$FRAMES")"
echo "  logs   : $LOG_DIR"

# ---------------------------------------------------------------- L1
head_ "L1  model chain (ONNX -> .rknn)"

if [ -f "$BACKBONE" ]; then
    run split "$PY" -m split_model --verify
    if [ $LAST_RC -eq 0 ]; then
        v="$(number "$LAST_LOG" '^gaze +:' 'max abs diff +[0-9.e+-]+')"
        if lt "$v" 1e-4; then
            pass "backbone split parity" "gaze diff ${v} rad"
        else
            fail "backbone split parity" "gaze diff ${v:-?} rad"
        fi
    elif grep -q "No module named 'onnxruntime'" "$LAST_LOG"; then
        skip "backbone split parity" "onnxruntime not installed"
    else
        fail "backbone split parity" "exit $LAST_RC"
    fi
else
    skip "backbone split parity" "missing $(basename "$BACKBONE")"
fi

if [ -f "$RKNN" ]; then
    pass "RKNN model present" "$(du -h "$RKNN" | cut -f1)"
else
    fail "RKNN model present" "missing $RKNN (run export_rknn.py)"
fi

# ---------------------------------------------------------------- L2
head_ "L2  NPU layer (RK3576)"

if [ -f "$RKNN" ]; then
    run verify_rknn "$PY" -m rknn_backend.verify --rknn "$RKNN"
    if [ $LAST_RC -eq 0 ]; then
        if grep -q "^\[SKIP\]" "$LAST_LOG"; then
            skip "NPU vs reference model" "$(grep -m1 '^\[SKIP\]' "$LAST_LOG")"
        else
            v="$(number "$LAST_LOG" 'max \|dPitch\|' 'max \|dPitch\| +[0-9.]+')"
            if lt "$v" 1.0; then
                pass "NPU vs reference model" "max |dPitch| ${v} deg"
            else
                fail "NPU vs reference model" "max |dPitch| ${v:-?} deg"
            fi
        fi
    else
        fail "NPU vs reference model" "exit $LAST_RC"
    fi

    run infer_rknn "$PY" infer.py --rknn "$RKNN" \
        --output "$LOG_DIR/gaze_result.png"
    if [ $LAST_RC -eq 0 ] && grep -q "gaze pitch" "$LAST_LOG"; then
        v="$(number "$LAST_LOG" '^gaze pitch' '\([0-9.-]+ deg\)')"
        pass "infer.py (single image)" "pitch ${v:-?}"
    else
        fail "infer.py (single image)" "exit $LAST_RC"
    fi

    run infer_crop "$PY" infer.py --auto-crop --rknn "$RKNN" \
        --image assets/portrait.jpg --output "$LOG_DIR/gaze_image.png"
    if [ $LAST_RC -eq 0 ] && grep -q "eye agreement" "$LAST_LOG"; then
        v="$(number "$LAST_LOG" 'eye agreement' 'agreement: +[0-9.]+')"
        if lt "$v" 5.0; then
            pass "infer.py --auto-crop (both eyes)" "eyes agree ${v:-?} deg"
        else
            fail "infer.py --auto-crop (both eyes)" "eyes disagree ${v:-?} deg"
        fi
    else
        fail "infer.py --auto-crop (both eyes)" "exit $LAST_RC"
    fi
else
    skip "NPU vs reference model" "no RKNN model"
    skip "infer.py (single image)" "no RKNN model"
    skip "infer.py --auto-crop (both eyes)" "no RKNN model"
fi

if [ "$RUN_QUICK" -eq 0 ] && [ -f "$RKNN" ]; then
    run video_rknn "$PY" video_infer.py --rknn "$RKNN" --max-frames "$FRAMES" \
        --output "$LOG_DIR/gaze_video.mp4" --csv "$LOG_DIR/gaze.csv"
    if [ $LAST_RC -eq 0 ] && [ -f "$LOG_DIR/gaze.csv" ]; then
        rows=$(( $(wc -l < "$LOG_DIR/gaze.csv") - 1 ))
        fps="$(grep -oE '\([0-9.]+ fps\)' "$LAST_LOG" | head -1)"
        pass "video_infer.py (NPU)" "${rows} frames ${fps}"
    else
        fail "video_infer.py (NPU)" "exit $LAST_RC"
    fi
else
    skip "video_infer.py (NPU)" "--quick or no RKNN model"
fi

# ---------------------------------------------------------------- summary
printf "\n== summary ==\n"
printf "  PASS=%d  FAIL=%d  SKIP=%d\n" "$PASS" "$FAIL" "$SKIP"
printf "  logs: %s\n" "$LOG_DIR"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
