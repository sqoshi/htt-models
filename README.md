<a id="readme-top"></a>
<div align="center">

[![Contributors](https://img.shields.io/github/contributors/sqoshi/htt-models.svg)](https://github.com/sqoshi/htt-models/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/sqoshi/htt-models.svg)](https://github.com/sqoshi/htt-models/network/members)
[![Stargazers](https://img.shields.io/github/stars/sqoshi/htt-models.svg)](https://github.com/sqoshi/htt-models/stargazers)
[![Issues](https://img.shields.io/github/issues/sqoshi/htt-models.svg)](https://github.com/sqoshi/htt-models/issues)

</div>

<br />
<div align="center">
  <a href="https://github.com/sqoshi/hands-to-text/blob/master/docs/landscape.png">
   <img src="https://github.com/sqoshi/hands-to-text/raw/master/docs/landscape.png" alt="Logo" width="720" height="320">
 </a>

<h3 align="center">htt-models</h3>

  <p align="center">
    Python package designed to train models and process data for hand gesture recognition.
    <br />
    <a href="https://github.com/sqoshi/htt-models"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sqoshi/htt-models">View Demo</a>
    &middot;
    <a href="https://github.com/sqoshi/htt-models/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/sqoshi/htt-models/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>

## About The Project

The htt-models package facilitates the training, processing, and versioning of machine learning models required for the Hands-to-Text application, which interprets hand gestures in real-time.

This package enables training models on hand gesture datasets such as MNIST and ASL Hands, providing state-of-the-art recognition of American Sign Language (ASL) letters. It includes multiple model implementations to ensure flexibility and robustness in recognition accuracy.

Additionally, trained models are packaged as pre-built Docker images, allowing seamless deployment across different environments, ensuring portability and ease of use.

## Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [scikit-learn](https://scikit-learn.org/) - Machine learning library
* [OpenCV](https://opencv.org/) - Computer vision library
* [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
* [Poetry](https://python-poetry.org/) - Dependency management

## Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites

* Python 3.10+
* Poetry

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/sqoshi/htt-models.git
   cd htt-models
   ```

2. Install dependencies
   ```sh
   poetry install
   ```

3. Download datasets
   ```sh
   mkdir -p data/mnist
   # Download MNIST dataset (will be downloaded automatically when first needed)

   # For ASL dataset, download from source and place in appropriate directory
   # or use existing dataset path with --data-path option
   ```

## Usage

### Command-Line Interface

The package provides a simple command-line interface for training and evaluating models:

```sh
# Train all models on all datasets
python -m httmodels.main --model all --dataset all

# Train specific model on specific dataset
python -m httmodels.main --model lenet --dataset mnist --epochs 10 --batch_size 64

# Train ResNet on ASL dataset with custom data path
python -m httmodels.main --model resnet --dataset asl --data_path /path/to/asl/dataset --epochs 15

# Train Random Forest on ASL dataset
python -m httmodels.main --model rf --dataset asl
```

### Training Options

The main script supports the following parameters:

- `--model`: Model to train (lenet, resnet, rf, adaboost, all)
- `--dataset`: Dataset to use (mnist, asl, all)
- `--data_path`: Path to the dataset directory
- `--epochs`: Number of epochs for neural network training
- `--batch_size`: Batch size for neural network training

# Run only the trainer examples
python run_examples.py --example trainers

# Specify a custom data path
python run_examples.py --data-path /path/to/datasets
```

### Using the Dataloader API

The package provides a streamlined dataloader API for working with datasets:

```python
from httmodels.dataloaders import (
    get_dataloader,
    get_mnist_transforms,
    create_train_val_dataloaders
)
from httmodels.preprocessing.mnist import MNISTProcessor

# Load dataset through processor
processor = MNISTProcessor(apply_augmentation=True)
data = processor.load()

# Get training and testing datasets
train_dataset = data["train"]
test_dataset = data["test"]

# Apply transforms
train_dataset.transform = get_mnist_transforms(augmentation=True)
test_dataset.transform = get_mnist_transforms(augmentation=False)

# Create dataloaders
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

# Or create train/val split from a single dataset
train_loader, val_loader = create_train_val_dataloaders(
    dataset=train_dataset,
    batch_size=64,
    train_ratio=0.8
)
```

### Using the Training API

Models can be trained using the trainer and context classes:

```python
from httmodels.trainers.lenet import LeNetTrainer
from httmodels.trainers.context import TrainingContext

# Initialize trainer
trainer = LeNetTrainer(input_shape=(1, 28, 28), num_classes=10)

# Train with context
context = TrainingContext(trainer)
context.fit(x_train, y_train, epochs=10, batch_size=64)
accuracy = context.evaluate(x_test, y_test)

# Save model
context.save("lenet_mnist.pth")
```

## Project Structure

```
httmodels/
├── config.py                  # Configuration settings
├── models/                    # Model implementations
│   ├── base.py                # Base model interfaces
│   ├── lenet.py               # LeNet CNN model
│   ├── resnet.py              # ResNet model
│   ├── random_forest.py       # Random Forest model
│   └── adaboost.py            # AdaBoost model
├── trainers/                  # Model trainers
│   ├── base.py                # Base trainer interfaces
│   ├── lenet.py               # LeNet trainer
│   ├── resnet.py              # ResNet trainer
│   ├── rf.py                  # Random Forest trainer
│   ├── adaboost.py            # AdaBoost trainer
│   └── context.py             # Training context
├── preprocessing/             # Data preprocessing
│   ├── base.py                # Base preprocessor interfaces
│   ├── mnist.py               # MNIST dataset preprocessor
│   └── aslhands.py            # ASL hands dataset preprocessor
├── datasets/                  # Dataset classes
│   └── datasets.py            # Unified MNIST and ASL dataset classes
├── dataloaders/               # Data loaders
│   ├── loaders.py             # Dataloader utilities and transforms
│   └── landmarkstransformers.py # MediaPipe hand landmark extractors
├── examples/                  # Example implementations
│   ├── dataloaders_example.py # Examples using the dataloader API
│   └── updated_trainers_example.py # Examples using the updated trainers
└── utils/                     # Utility functions
    ├── common.py              # Common utility functions
    └── ...
```

## Roadmap

- [x] Base model and trainer interfaces
- [x] CNN models (LeNet, ResNet)
- [x] ML models (Random Forest, AdaBoost)
- [x] Data preprocessing for MNIST and ASL
- [x] Unified dataloaders and transforms
- [ ] Add support for more datasets
- [ ] Implement more model architectures
- [ ] Add model serving capabilities
- [ ] Improve documentation and examples

See the [open issues](https://github.com/sqoshi/htt-models/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Python
* NumPy
* Scikit-Learn
* Pandas
* PyTorch
* TorchVision
* TorchViz
* OpenCV
* Pydantic
* MediaPipe

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get started with htt-models, follow these steps:

### Installation

Package installation via pip artifactory:

```sh
    pipx install htt-models
```

Pretrained models download via docker:

```dockerfile
    FROM ghcr.io/sqoshi/htt-models:latest AS models
    COPY --from=models /models /models
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

* [x] Supports Multiple Datasets
  * [x] MNIST - Standard benchmark dataset for digit recognition.
  * [x] ASL Hands - Dataset specifically designed for ASL gesture recognition.

* [x] Prebuilt Model Trainers
  * [x] ResNet-18 - Deep learning model using Residual Networks.
  * [x] LeNet - Classic Convolutional Neural Network (CNN) for image classification.
  * [x] Random Forest - Efficient tree-based ensemble method for classification tasks.
  * [x] AdaBoost - Boosting algorithm improving accuracy with ensemble weak learners.

* [x] Flexible Data Loaders
  * [x] ResNet18 DataLoader - Loads data for deep learning models.
  * [x] LeNet DataLoader - Standard data pipeline for CNN-based training.
  * [x] RandomForest DataLoader - Optimized for tree-based classification models.

* [x] Seamless Deployment
  * [x] Pre-trained models are containerized in Docker, ensuring easy deployment across different systems.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
