#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfg_files/fit_smplx_own.yaml}"
INPUT_DIR="${2:-own_skeletons}"
OUTPUT_DIR="${3:-output_folder}"

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

for csv_path in "${csv_files[@]}"; do
    sample_name="$(basename "$csv_path" .csv)"
    result_json="$OUTPUT_DIR/$sample_name/body_smplx.json"

    if [[ -f "$result_json" ]]; then
        echo "Skipping $csv_path because $result_json already exists"
        continue
    fi

    echo "========================================"
    echo "Fitting $csv_path"
    echo "========================================"
    python main.py \
        --config "$CONFIG" \
        --dataset own \
        --data_folder "$csv_path" \
        --output_folder "$OUTPUT_DIR"
done

echo "Batch fitting complete."
