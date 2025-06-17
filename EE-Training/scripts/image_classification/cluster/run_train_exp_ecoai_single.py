# Author: Bhavesh Wani
# This script is used to run a single training experiment on the cluster and track energy with eco2ai and DCGM.

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys, subprocess
from eco2ai import Tracker
import warnings
warnings.filterwarnings("ignore")


# === Experiment settings ===
model = sys.argv[1]
dataset_name = sys.argv[2]
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])

# dataset_flag = "torch/image_folder"
dataset_flag = "torch/msgpack"


gpus = sys.argv[5]
experiment_name = f"{model}_{dataset_name}_{epochs}ep_bs{batch_size}"


# === Paths === 
base_output = f"/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/{gpus}"
output_path = f"{base_output}/models/{experiment_name}"
eco2ai_log_dir = f"{base_output}/eco2ai_logs"
train_script = "/netscratch/bwani/pytorch-image-models/train.py"

# data_dir = "/netscratch/bwani/pytorch-image-models/ds/imagenet"
data_dir = "/ds/images/imagenet/msgpack"



# Create necessary directories
os.makedirs(base_output, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(eco2ai_log_dir, exist_ok=True)

eco2ai_log_file = os.path.join(eco2ai_log_dir, f"{experiment_name}.csv")


# === Start eco2AI tracker ===
tracker = Tracker(
    project_name=experiment_name,
    experiment_description=experiment_name,
    file_name=eco2ai_log_file
)


# === Start training ===
tracker.start()
try:
    subprocess.run([
        "python", train_script,
        "--model", model,
        "--dataset", dataset_flag,
        "--data-dir", data_dir,
        "--dataset-download",
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--output", output_path,
        "--experiment", experiment_name,
        "--amp",
        "--workers", "8",
        "--pin-mem"
    ])
finally:
    tracker.stop()

print(f"✅ Finished: {experiment_name}")