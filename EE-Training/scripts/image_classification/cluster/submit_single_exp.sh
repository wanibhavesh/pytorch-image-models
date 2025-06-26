#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --time=3-00:00:00                   
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_24.11-py3.sqsh
#SBATCH --container-mounts="/netscratch/bwani/pytorch-image-models:/netscratch/bwani/pytorch-image-models,/ds/images/imagenet:/ds/images/imagenet"
#SBATCH --container-workdir="/netscratch/bwani/pytorch-image-models"


# === Parse input arguments ===
MODEL=$1
DATASET=$2
EPOCHS=$3
BS=$4
GPU=$5
DATA_PERCENT=${6:-1.0}  # Default to 1.0 if not provided

if (( $(echo "$DATA_PERCENT < 1.0" | bc -l) )); then
    EXP_NAME="escade_${MODEL}_ep${EPOCHS}_bs${BS}_data$(echo "$DATA_PERCENT * 100" | bc | cut -d. -f1)pct"
else
    EXP_NAME="escade_${MODEL}_ep${EPOCHS}_bs${BS}"
fi

GPU_LOG="/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/${GPU}/dcgm_logs/${EXP_NAME}.csv"


# === Start DCGM GPU logger in background ===
bash EE-Training/scripts/image_classification/cluster/gpu_logger.sh "$GPU_LOG" &
LOGGER_PID=$!

# === Execute the training script with parameters ===
./install.sh python EE-Training/scripts/image_classification/cluster/run_train_exp_ecoai_single.py "$1" "$2" "$3" "$4" "$5" "$DATA_PERCENT"

# === Stop GPU logger ===
kill $LOGGER_PID
wait $LOGGER_PID 2>/dev/null

echo "✅ Training finished: $EXP_NAME"