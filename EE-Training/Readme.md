# Energy-Efficient Training

---

## Repository Setup

```bash
EE-Training/
├── Dataset                          # Energy consumption data
├── scirpts                          # Code for running training
└── README.md
```

## Image Classification

This directory (`scripts/image_classification`) contains code for training image classification models using [`pytorch-image-models`](https://github.com/huggingface/pytorch-image-models), with energy consumption data recorded using [CodeCarbon](https://github.com/mlco2/codecarbon).

Clone the official Timm training repo:

```bash
git clone https://github.com/huggingface/pytorch-image-models.git
```

### Experiment Summary

Total of **195 energy consumption experiments** - 14 **models**, **epochs** (50, 75, 100), **datasets** (CIFAR-10, CIFAR-100), and **batch sizes** (32, 64, 128).

### Models

| Model                    | Epochs           | Datasets               | Batch Sizes     |
|-------------------------|------------------|------------------------|-----------------|
| convnext_tiny           | [100]            | [cifar10]              | [64, 128]       |
| deit_base_patch16_224   | [100]            | [cifar10]              | [64, 128]       |
| efficientnet_b0         | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| efficientnet_b3         | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| efficientnetv2_s        | [50, 75, 100]    | [cifar10]              | [32, 64, 128]   |
| mobilenetv2_100         | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| mobilenetv3_large_100   | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| mobilenetv3_small_100   | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| resnet101               | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| resnet152               | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| resnet18                | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| resnet34                | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| resnet50                | [50, 75, 100]    | [cifar10, cifar100]    | [32, 64, 128]   |
| vit_base_patch16_224    | [100]            | [cifar10]              | [64, 128]       |


> Full experiment logs `Dataset/image_classification/training_energy.csv`.


### Parameters

The following parameters are recorded for each experiment:

| Parameter         | Description                                                   |
|------------------|---------------------------------------------------------------|
| timestamp         |                                                               |
| project_name      | CodeCarbon project                                            |
| run_id            | Unique run identifier                                         |
| experiment_id     | Experiment ID                                                 |
| experiment_name   | Experiment label                                              |
| model             | Model                                                         |
| dataset           | Dataset (CIFAR-10, CIFAR-100)                                 |
| epochs            | Number of training epochs                                     |
| batch_size        | Batch size used during training                               |
| duration          | Duration of the training (in seconds)                         |
| emissions         | Total CO₂ emissions in kilograms                              |
| emissions_rate    | CO₂ emissions rate (kg/sec)                                   |
| cpu_power         | Average CPU power usage (W)                                   |
| gpu_power         | Average GPU power usage (W)                                   |
| ram_power         | Average RAM power usage (W)                                   |
| cpu_energy        | Total CPU energy consumed (kWh)                               |
| gpu_energy        | Total GPU energy consumed (kWh)                               |
| ram_energy        | Total RAM energy consumed (kWh)                               |
| energy_consumed   | Total energy consumed (kWh)                                   |
| country_name      | Country                                                       |
| country_iso_code  | ISO code of the country                                       |
| region            | Region                                                        |
| cloud_provider    | Cloud provider (if any)                                  |
| cloud_region      | Cloud region (if any)                                  |
| os                | Operating system                                               |
| python_version    | Python version used                                            |
| codecarbon_version| CodeCarbon version                                             |
| cpu_count         | Number of CPU cores                                            |
| cpu_model         | CPU model name                                                 |
| gpu_count         | Number of GPUs                                                 |
| gpu_model         | GPU model name                                                 |
| longitude         | Geo longitude (approximate)                                    |
| latitude          | Geo latitude (approximate)                                     |
| ram_total_size    | Total RAM available (in GB)                                    |
| tracking_mode     | CodeCarbon tracking mode (process, machine, etc.)             |
| on_cloud          | Whether exp was on a cloud machine                             |
| pue               | Power Usage Effectiveness value                                |
| train_loss        | Final training loss                                            |
| eval_loss         | Final evaluation loss                                          |
| eval_top1         | Top-1 evaluation accuracy (%)                                  |
| eval_top5         | Top-5 evaluation accuracy (%)                                  |
| lr                | Learning rate used                                             |

