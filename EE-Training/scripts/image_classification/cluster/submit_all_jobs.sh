#!/bin/bash

GPU_TYPE="A100-80GB"

MODELS=(
    "resnetv2_152x2_bit"
)

DATASET="imagenet-1k"
EPOCHS=(75)
BATCH_SIZES=(256)



for model in "${MODELS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      exp_name="escade_${model}_ep${epoch}_bs${bs}"
      log_dir="/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/logs/${GPU_TYPE}"
      mkdir -p "$log_dir"

      sbatch \
        --partition="$GPU_TYPE" \
        --job-name="$exp_name" \
        --output="${log_dir}/${exp_name}_%j.log" \
        --error="${log_dir}/${exp_name}_%j.log" \
        submit_single_exp.sh "$model" "$DATASET" "$epoch" "$bs" "$GPU_TYPE"

      echo "📤 Submitted job for $exp_name"
    done
  done
done

