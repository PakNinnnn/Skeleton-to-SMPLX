#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output_folder}"
CSV_DIR="${2:-/home/marcolee/files/badminton/tsad/EDA/processed_skeleton/sub1}"
FRAME_START="${3:-0}"
FRAME_END="${4:-}"

FPS="${FPS:-10}"
MP4_ELEV="${MP4_ELEV:-51.749}"
MP4_AZIM="${MP4_AZIM:-153.069}"
MP4_ROLL="${MP4_ROLL:-150.416}"
ROTATE_CCW="${ROTATE_CCW:-true}"
SHOW_EDGES="${SHOW_EDGES:-false}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Output directory not found: $OUTPUT_DIR" >&2
    exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
    echo "CSV directory not found: $CSV_DIR" >&2
    exit 1
fi

shopt -s nullglob
sample_dirs=("$OUTPUT_DIR"/*)
shopt -u nullglob

if [[ ${#sample_dirs[@]} -eq 0 ]]; then
    echo "No items found in: $OUTPUT_DIR" >&2
    exit 1
fi

frame_range_requested=false
if [[ $# -ge 3 ]]; then
    frame_range_requested=true
fi

echo "Scanning ${#sample_dirs[@]} item(s) under $OUTPUT_DIR"
echo "Matching CSV files from: $CSV_DIR"

processed=0
skipped=0

for sample_dir in "${sample_dirs[@]}"; do
    if [[ ! -d "$sample_dir" ]]; then
        continue
    fi

    sample_name="$(basename "$sample_dir")"
    csv_path="$CSV_DIR/$sample_name.csv"
    mesh_dir="$sample_dir/meshes"
    body_json="$sample_dir/body_smplx.json"
    output_html="$sample_dir/viewer.html"
    output_mp4="$sample_dir/smplx.mp4"

    if [[ ! -f "$csv_path" ]]; then
        echo "Skipping $sample_name: CSV not found at $csv_path"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -d "$mesh_dir" ]]; then
        echo "Skipping $sample_name: mesh directory not found at $mesh_dir"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -f "$body_json" ]]; then
        echo "Skipping $sample_name: body_smplx.json not found at $body_json"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -f "$output_mp4" && "$OVERWRITE" != "true" && "$frame_range_requested" != "true" ]]; then
        echo "Skipping $sample_name: $output_mp4 already exists"
        skipped=$((skipped + 1))
        continue
    fi

    args=(
        python visualize_own.py
        --skeleton_csv "$csv_path"
        --mesh_dir "$mesh_dir"
        --output_html "$output_html"
        --output_mp4 "$output_mp4"
        --frame_start "$FRAME_START"
        --fps "$FPS"
        --mp4_elev "$MP4_ELEV"
        --mp4_azim "$MP4_AZIM"
        --mp4_roll "$MP4_ROLL"
    )

    if [[ -n "$FRAME_END" ]]; then
        args+=(--frame_end "$FRAME_END")
    fi

    if [[ "$ROTATE_CCW" == "true" ]]; then
        args+=(--mp4_rotate_ccw)
    fi

    # if [[ "$SHOW_EDGES" == "true" ]]; then
    #     args+=(--mp4_show_edges)
    # fi

    echo "========================================"
    echo "Visualizing $sample_name"
    echo "CSV:   $csv_path"
    echo "Mesh:  $mesh_dir"
    echo "MP4:   $output_mp4"
    echo "HTML:  $output_html"
    echo "========================================"
    "${args[@]}"
    processed=$((processed + 1))
done

echo "Batch visualization complete. Processed: $processed, skipped: $skipped."
