#!/bin/bash

python visualize_own.py \
    --skeleton_csv /home/marcolee/files/badminton/tsad/EDA/processed_skeleton/sub1/_sub01_beginner_processed_290.csv \
    --mesh_dir /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_290/meshes \
    --output_html /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_290/viewer.html \
    --output_mp4 /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_290/smplx.mp4 \
    --fps 10 \
    --mp4_elev 51.749 \
    --mp4_azim 153.069 \
    --mp4_roll 150.416 \
    --mp4_rotate_ccw
