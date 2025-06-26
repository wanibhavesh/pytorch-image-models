#!/bin/bash

GPU_TYPE="A100-80GB"

MODELS=(
    "resnetv2_152x2_bit"
)

DATASET="imagenet-1k"
EPOCHS=(75)
BATCH_SIZES=(256)
DATA_PERCENTAGES=(1.0 0.5 0.25 0.1)  # Add different data percentages to test



for model in "${MODELS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for data_pct in "${DATA_PERCENTAGES[@]}"; do
        if (( $(echo "$data_pct < 1.0" | bc -l) )); then
          exp_name="escade_${model}_ep${epoch}_bs${bs}_data$(echo "$data_pct * 100" | bc | cut -d. -f1)pct"
        else
          exp_name="escade_${model}_ep${epoch}_bs${bs}"
        fi
        
        log_dir="/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/logs/${GPU_TYPE}"
        mkdir -p "$log_dir"

        sbatch \
          --partition="$GPU_TYPE" \
          --job-name="$exp_name" \
          --output="${log_dir}/${exp_name}_%j.log" \
          --error="${log_dir}/${exp_name}_%j.log" \
          submit_single_exp.sh "$model" "$DATASET" "$epoch" "$bs" "$GPU_TYPE" "$data_pct"

        echo "📤 Submitted job for $exp_name (data: ${data_pct})"
      done
    done
  done
done

