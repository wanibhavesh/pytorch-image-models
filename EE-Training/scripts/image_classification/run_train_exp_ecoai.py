import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import subprocess
from eco2ai import Tracker
import warnings
warnings.filterwarnings("ignore")


# === Short Exp setup ===
gpus = "A100-80"
models = [
    "resnet50"
]
datasets = [
    ("torch/image_folder", "imagenet-1k")
]
epochs_list = [10]
batch_sizes = [128]




# === Experiment settings ===

# models = [
#     "deit_base_patch16_224",
#     "resnet152",
#     "vit_base_patch16_224"
# ]

# datasets = [
#     ("torch/imagenet", "imagenet-1k")
# ]


#   == GPU settings ===
# Note: The following GPUs are used for training. Adjust as per your environment.
# A100-80: 80GB A100 GPU
# H100: 80GB H100 GPU
# H200: 80GB H200 GPU
# These GPUs are used for training. Adjust as per your environment.

# gpus = "A100-80"

# epochs_list = [300]
# batch_sizes = [128, 224, 512]


# === Paths ===
# Cluster-specific paths

# base_output = "/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/models/{gpus}"
# data_dir = "/ds/images"
# eco2ai_log_dir = "/netscratch/bwani/pytorch-image-models/EE-Training/evaluation/{gpus}/eco2ai_logs"
# train_script = "/netscratch/bwani/pytorch-image-models/EE-Training/scripts/image_classification/train.py"


# Local paths for testing purposes
base_output = f"/home/escade/ESCADE/code/git/pytorch-image-models/EE-Training/evaluation/{gpus}"
data_dir = "/home/escade/ESCADE/Dataset/imagenet"
eco2ai_log_dir = f"/home/escade/ESCADE/code/git/pytorch-image-models/EE-Training/evaluation/{gpus}/eco2ai_logs"
train_script = "/home/escade/ESCADE/code/git/pytorch-image-models/train.py"

os.makedirs(base_output, exist_ok=True)
os.makedirs(eco2ai_log_dir, exist_ok=True)

# === Training loop ===
for model in models:
    for (dataset_flag, dataset_name) in datasets:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                experiment_name = f"{model}_{dataset_name}_{epochs}ep_bs{batch_size}"
                output_path = os.path.join(base_output, experiment_name)
                eco2ai_log_file = os.path.join(eco2ai_log_dir, f"{experiment_name}.csv")

                print(f"🚀 Starting: {experiment_name}")
                os.makedirs(output_path, exist_ok=True)

                tracker = Tracker(
                    project_name=f"{experiment_name}",
                    experiment_description=f"{experiment_name}",
                    file_name=eco2ai_log_file
                )

                tracker.start()
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
                    "--amp"
                ])
                tracker.stop()

                print(f"✅ Completed: {experiment_name} — log saved to {eco2ai_log_file}")
