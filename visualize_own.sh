#!/bin/bash

# python visualize_own.py \
#     --skeleton_csv /home/marcolee/files/badminton/tsad/EDA/processed_skeleton/sub11/_sub11_pro_processed_334.csv \
#     --mesh_dir /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub11_pro_processed_334/meshes \
#     --output_html /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub11_pro_processed_334/viewer.html \
#     --output_mp4 /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub11_pro_processed_334/smplx_follow.mp4 \
#     --fps 10 \
#     --mp4_elev 51.749 \
#     --mp4_azim 153.069 \
#     --mp4_roll 150.416 \
#     --mp4_rotate_ccw \
#     --frame_start 42 \
#     --frame_end 151 \
#     --mp4_label expert 

python visualize_own.py \
    --skeleton_csv /home/marcolee/files/badminton/tsad/EDA/processed_skeleton/sub1/_sub01_beginner_processed_302.csv \
    --mesh_dir /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_302/meshes \
    --output_html /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_302/viewer.html \
    --output_mp4 /home/marcolee/files/badminton/smplifyx-skeleton/output_folder/_sub01_beginner_processed_302/smplx_follow.mp4 \
    --fps 10 \
    --mp4_elev 51.749 \
    --mp4_azim 153.069 \
    --mp4_roll 150.416 \
    --mp4_rotate_ccw \
    --frame_start 53 \
    --frame_end 151 \
    --mp4_label player 
