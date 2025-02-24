#!/bin/bash
set -e

# Define your working directory, model, and checkpoint names.
workdir='.'
model_name='ours'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

# Define the output directory and the directory containing your raw videos.
output_dir="/nas/spatial/source_datasets/scannet/datasets/scans_videos"
video_dir="/nas/spatial/source_datasets/scannet/datasets/scans_videos"   # update this path to point to your video folder
num_frames=32

# Ensure PYTHONPATH includes your project root, src, and eval directories.
export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/eval"

echo "Output directory: ${output_dir}"
echo "Model weights: ${model_weights}"
echo "Video directory: ${video_dir}"
echo "Number of frames per video: ${num_frames}"

# Launch the feature extraction script using Accelerate.
# If feature extraction is single-process, you may use --num_processes 1.
# If you want to distribute across multiple processes, adjust accordingly.
accelerate launch --num_processes 1 /home/rilyn/CUT3R/eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --video_dir "$video_dir" \
    --num_frames "$num_frames" \
    --model_name "$model_name"
