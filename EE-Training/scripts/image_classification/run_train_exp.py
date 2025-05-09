import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import csv
import subprocess
from codecarbon import EmissionsTracker
import pandas as pd

# experiment settings
models = [
    "convnext_tiny",
    "deit_base_patch16_224",
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnetv2_s",
    "mobilenetv2_100",
    "mobilenetv3_large_100",
    "mobilenetv3_small_100",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "vit_base_patch16_224"
]

datasets = [
    ("torch/cifar10", "cifar10"),
    ("torch/cifar100", "cifar100"),
]


epochs_list = [50, 75, 100]
batch_sizes = [32, 64, 128]


# Change the Path before running

base_output = "/home/escade/ESCADE/code/git/pytorch-image-models/EE-Training/Evaluation/exp_models"
log_file = "/home/escade/ESCADE/code/git/pytorch-image-models/EE-Training/Evaluation/energy_results.csv"
data_dir = "/home/escade/ESCADE/code/EE_Training/pytorch-image-models/data"
codecarbon_log_dir = "/home/escade/ESCADE/code/git/pytorch-image-models/EE-Training/Evaluation/codecarbon_logs"


# Create necessary directories
os.makedirs(base_output, exist_ok=True)
os.makedirs(codecarbon_log_dir, exist_ok=True)

# Load completed experiments
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    completed_experiments = set(df["experiment_name"])
else:
    completed_experiments = set()

# Run training loop
for model in models:
    for (dataset_flag, dataset_name) in datasets:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                experiment_name = f"{model}_{dataset_name}_{epochs}ep_bs{batch_size}"
                output_path = os.path.join(base_output, experiment_name)

                if experiment_name in completed_experiments:
                    print(f"Skipping {experiment_name}, already exists.")
                    continue

                print(f"Starting: {experiment_name}")

                tracker = EmissionsTracker(
                    project_name=experiment_name,
                    output_file=f"{experiment_name}.csv",
                    output_dir=codecarbon_log_dir
                )
                tracker.start()

                # Run model training
                subprocess.run([
                    "python", "/home/escade/ESCADE/code/git/pytorch-image-models/train.py",
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
                emissions = tracker._prepare_emissions_data()

                
                # Prepare combined row
                row = emissions.__dict__.copy()
                row.update({
                    "experiment_name": experiment_name,
                    "model": model,
                    "dataset": dataset_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                })

                # Write to Master CSV
                fieldnames = list(row.keys())
                write_header = not os.path.exists(log_file)
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                print(f"Completed: {experiment_name} — {emissions.energy_consumed:.6f} kWh")
