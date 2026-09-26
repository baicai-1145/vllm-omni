#!/bin/bash
# 用法: ./gen_sample.sh [port] [outfile] [seed] [steps] [duration] [width] [height]
# 官方 T2VA multipart API: POST /v1/videos/sync → mp4 字节
set -u
PORT=${1:-9098}
OUT=${2:-/workspace/.tmp/h3-mission/samples/gen.mp4}
SEED=${3:-1101}
STEPS=${4:-50}
DUR=${5:-5.0}
W=${6:-1344}
H=${7:-768}
M=/workspace/.tmp/h3-mission
PROMPT="A golden retriever sprinting through a sunlit meadow, cinematic slow motion, dust motes glowing in warm backlight, shallow depth of field, 35mm film look."
mkdir -p "$(dirname "$OUT")"

# Turbo merged (steps<=8): shift 12/3, no request lora; FlashGen via FLASHGEN=1; else plain
LORA_ARGS=()
if [ "$STEPS" -le 8 ] && [ "${FLASHGEN:-0}" = "1" ]; then
  LORA_FILE=$M/flashgen-lora/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
  LORA_ARGS=(-F "lora={\"name\":\"h3-flashgen-v1.0\",\"path\":\"${LORA_FILE}\",\"scale\":1.0}")
else
  LORA_ARGS=(-F "flow_shift=12.0" -F "audio_flow_shift=3.0")
fi

START=$(date +%s.%N)
HDR=$(mktemp)
http=$(curl -sS -D "$HDR" -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F "prompt=${PROMPT}" \
  -F "width=${W}" \
  -F "height=${H}" \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F "num_inference_steps=${STEPS}" \
  "${LORA_ARGS[@]}" \
  -F "seed=${SEED}" \
  -F "extra_params={\"task\":\"t2va\",\"duration\":${DUR}}" \
  -o "$OUT" -w "%{http_code}")
END=$(date +%s.%N)
WALL=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')

if [ "$http" = "200" ] && [ -s "$OUT" ]; then
  echo "GEN_OK $OUT wall=${WALL}s http=$http size=$(stat -c%s "$OUT")"
else
  echo "GEN_FAIL http=$http wall=${WALL}s"
  head -c 400 "$OUT" 2>/dev/null
  exit 1
fi
