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
    Python package designed to train models and process data.
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
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>

## About The Project

The htt-models package facilitates the training, processing, and versioning of machine learning models required for the Hands-to-Text application, which interprets hand gestures in real-time.

This package enables training models on hand gesture datasets such as MNIST and ASL Hands, providing state-of-the-art recognition of American Sign Language (ASL) letters. It includes multiple model implementations to ensure flexibility and robustness in recognition accuracy.

Additionally, trained models are packaged as pre-built Docker images, allowing seamless deployment across different environments, ensuring portability and ease of use.

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
