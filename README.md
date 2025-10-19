# Face Recognition Models

This repository provides PyTorch implementations of several state-of-the-art deep learning models for face recognition. The project focuses on training models using large-margin cosine-based loss functions on the CASIA-WebFace dataset and evaluating their verification performance on the Labeled Faces in the Wild (LFW) dataset.

## Features

*   **Multiple Model Implementations**: Includes SphereFace, CosFace, ArcFace, CurricularFace, and a baseline FaceNet with Triplet Loss.
*   **Efficient Training Pipeline**: A robust training script utilizing mixed-precision (`torch.amp`) for faster training, flexible checkpointing to resume from the latest or best model state, and seamless integration with Weights & Biases (W&B) for experiment tracking.
*   **Standardized Evaluation**: Implements the standard 10-fold cross-validation protocol on the LFW dataset to benchmark model performance.
*   **Modular Design**: The codebase is organized with clear separation for datasets, loss functions (criteria), optimizers, and training utilities, making it easy to extend and experiment with new architectures.
*   **Flexible Configuration**: Utilizes a central configuration file for managing file paths and model hyperparameters.

## Models Implemented

The core of this repository lies in the `sphere-cos-arc` directory, which provides a unified framework for the following margin-based loss functions:

*   **SphereFace (`sphereface.py`)**: Implements the Angular Softmax (A-Softmax) loss, which introduces an angular margin to the target logit.
*   **CosFace (`cosface.py`)**: Implements the Large Margin Cosine Loss (LMCL), which adds a cosine margin directly to the target logit.
*   **ArcFace (`arcface.py`)**: Implements the Additive Angular Margin Loss, which adds an angular margin directly to the angle between the feature vector and the class center.
*   **CurricularFace (`curricular.py`)**: An adaptive curriculum learning approach that applies a margin based on the relative difficulty of training samples.
*   **FaceNet (`FaceNet/`)**: A separate implementation of the original FaceNet using Triplet Loss with semi-hard online triplet mining.

All models in the `sphere-cos-arc` framework use a pre-trained **ResNet-50** backbone by default.

## Repository Structure

```
.
├── FaceNet/                  # FaceNet implementation with Triplet Loss
│   ├── main.py               # Main script for model, sampler, and training
│   └── utils/
│       ├── criterions.py     # Triplet, CosFace, ArcFace loss definitions
│       └── dataset.py        # LFW dataset loader for triplets
│
└── sphere-cos-arc/           # Main framework for margin-based models
    ├── arcface.py            # Training script for ArcFace
    ├── cosface.py            # Training script for CosFace
    ├── curricular.py         # Training script for CurricularFace
    ├── sphereface.py         # Training script for SphereFace
    ├── run.sh                # Shell script to execute training runs
    └── utils/
        ├── config.py         # Central configuration for paths and hyperparameters
        ├── criterion.py      # Implementations of SphereFace, CosFace, ArcFace, etc.
        ├── dataset.py        # DataLoaders for CASIA-WebFace and LFW
        ├── model_utils.py    # Core training, evaluation, and checkpointing pipeline
        ├── optimizers.py     # Optimizer factory
        ├── schedulers.py     # Learning rate scheduler factory
        └── utils.py          # General utilities
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Weights & Biases (`wandb`)
*   `numpy`
*   `tqdm`
*   `python-dotenv`
*   `alive-progress`

You can install the dependencies via pip:
```bash
pip install torch torchvision numpy wandb tqdm python-dotenv alive-progress
```

### Dataset Setup

1.  **Training Data (CASIA-WebFace)**: Download the CASIA-WebFace dataset and extract it.
2.  **Evaluation Data (LFW)**: Download the Labeled Faces in the Wild (LFW) dataset and extract it.
3.  Ensure your final directory structure resembles the following:
    ```
    /path/to/your/datasets/
    ├── CASIA-webface/
    │   ├── 0000045/
    │   │   ├── 001.jpg
    │   │   └── ...
    │   └── ...
    └── Labeled Faces in the Wild (LFW)/
        ├── lfw_funneled/
        │   ├── Aaron_Eckhart/
        │   │   ├── Aaron_Eckhart_0001.jpg
        │   │   └── ...
        │   └── ...
        ├── pairs.txt
        ├── pairsDevTrain.txt
        └── pairsDevTest.txt
    ```

### Configuration

1.  **File Paths**: Open `sphere-cos-arc/utils/config.py` and update the `DATASET_PATH` and `WORKING_PATH` variables to point to your dataset directory and the repository's `sphere-cos-arc` directory, respectively.

    ```python
    # sphere-cos-arc/utils/config.py
    DATASET_PATH = "/path/to/your/datasets"
    WORKING_PATH = "/path/to/your/repo/sphere-cos-arc"
    CHECKPOINTS_FOLDER_PATH = "/path/to/your/repo/sphere-cos-arc/checkpoints"
    ```

2.  **Weights & Biases**: Create a `.env` file in the root of the repository and add your W&B API key.

    ```
    # .env
    WANDB_API_KEY="your_wandb_api_key_here"
    ```

## Training and Evaluation

The repository includes a convenience script `run.sh` to launch training jobs. The training process will automatically conclude with an evaluation on the LFW dataset, which includes tuning the verification threshold and performing 10-fold cross-validation.

To train a model, execute the `run.sh` script, passing the desired model script as the first argument, followed by any hyperparameter overrides.

### Examples

**1. Train ArcFace with default settings:**
```bash
bash sphere-cos-arc/run.sh sphere-cos-arc/arcface.py
```

**2. Train CosFace with a custom batch size and learning rate for 50 epochs:**
```bash
bash sphere-cos-arc/run.sh sphere-cos-arc/cosface.py --batch-size 256 --epochs 50 --lr 0.01
```

**3. Resume training for SphereFace from the latest checkpoint:**
```bash
bash sphere-cos-arc/run.sh sphere-cos-arc/sphereface.py --continue_train latest
```

**4. Resume training from the best-performing (minimum loss) checkpoint:**
```bash
bash sphere-cos-arc/run.sh sphere-cos-arc/sphereface.py --continue_train min_loss
```

All training logs, model checkpoints, and evaluation results will be logged to your Weights & Biases project and saved locally in the configured `log` and `checkpoints` directories.