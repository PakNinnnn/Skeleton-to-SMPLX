#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfg_files/fit_smplx_own.yaml}"
INPUT_DIR="${2:-/home/marcolee/files/badminton/tsad/EDA/convert_to_mesh}"
OUTPUT_DIR="${3:-output_folder}"
RETRY_FAILED="${RETRY_FAILED:-false}"
GPU_ID="${GPU_ID:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

shopt -s nullglob
csv_files=("$INPUT_DIR"/*.csv)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files found in: $INPUT_DIR" >&2
    exit 1
fi

echo "Found ${#csv_files[@]} CSV file(s) in $INPUT_DIR"
echo "Using config: $CONFIG"
echo "Writing outputs under: $OUTPUT_DIR"
echo "Retry failed samples: $RETRY_FAILED"
if [[ -n "$GPU_ID" ]]; then
    echo "Overriding GPU id: $GPU_ID"
fi

mkdir -p "$OUTPUT_DIR/.fit_locks" "$OUTPUT_DIR/.fit_logs"

processed=0
skipped=0
failed=0
current_lock=""

cleanup_current_lock() {
    if [[ -n "${current_lock:-}" && -d "$current_lock" ]]; then
        rmdir "$current_lock" 2>/dev/null || true
    fi
}

trap cleanup_current_lock EXIT INT TERM

for csv_path in "${csv_files[@]}"; do
    sample_name="$(basename "$csv_path" .csv)"
    sample_output_dir="$OUTPUT_DIR/$sample_name"
    result_json="$OUTPUT_DIR/$sample_name/body_smplx.json"
    failed_marker="$sample_output_dir/.failed"
    lock_dir="$OUTPUT_DIR/.fit_locks/$sample_name.lock"
    log_path="$OUTPUT_DIR/.fit_logs/$sample_name.log"

    if [[ -f "$result_json" ]]; then
        echo "Skipping $csv_path because $result_json already exists"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -f "$failed_marker" && "$RETRY_FAILED" != "true" ]]; then
        echo "Skipping $csv_path because previous failure marker exists: $failed_marker"
        skipped=$((skipped + 1))
        continue
    fi

    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "Skipping $csv_path because another process owns lock: $lock_dir"
        skipped=$((skipped + 1))
        continue
    fi
    current_lock="$lock_dir"

    echo "========================================"
    echo "Fitting $csv_path"
    echo "Lock: $lock_dir"
    echo "Log:  $log_path"
    echo "========================================"

    mkdir -p "$sample_output_dir"
    rm -f "$failed_marker"

    args=(
        python main.py
        --config "$CONFIG"
        --dataset own
        --data_folder "$csv_path"
        --output_folder "$OUTPUT_DIR"
    )

    if [[ -n "$GPU_ID" ]]; then
        args+=(--gpu_id "$GPU_ID")
    fi

    if "${args[@]}" 2>&1 | tee "$log_path"; then
        if [[ -f "$result_json" ]]; then
            echo "Finished $sample_name"
            processed=$((processed + 1))
        else
            echo "Command completed but result missing for $sample_name" | tee -a "$log_path"
            touch "$failed_marker"
            failed=$((failed + 1))
        fi
    else
        echo "Fitting failed for $sample_name" | tee -a "$log_path"
        touch "$failed_marker"
        failed=$((failed + 1))
    fi

    rmdir "$lock_dir"
    current_lock=""
done

echo "Batch fitting complete. Processed: $processed, skipped: $skipped, failed: $failed."
