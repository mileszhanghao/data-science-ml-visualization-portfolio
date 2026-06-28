# CIFAR-10 Deep Learning Experiments

PyTorch experiments comparing neural-network architectures for CIFAR-10 image classification.

## Experiment Design

The work compared a progression of models:

- Linear classifier baseline.
- Single-hidden-layer multilayer perceptron.
- Two-hidden-layer multilayer perceptron.
- Single-convolution CNN with max pooling.
- Deeper CNN with two convolution blocks and fully connected layers.
- CNN variant with batch normalization and dropout regularization.

## Training Workflow

- Used PyTorch data loaders for train and validation sets.
- Trained with cross-entropy loss and Adam optimization.
- Tracked training and validation accuracy across epochs.
- Compared architectural changes to reason about underfitting, overfitting, and regularization.

## Skills Demonstrated

- PyTorch model definition with `nn.Module`.
- Forward-pass implementation for MLP and CNN architectures.
- Accuracy evaluation loops.
- Batch normalization and dropout.
- Image classification workflow design.

## Publication Note

The original notebook contained course instructions and explicit redistribution restrictions, so it is not published here. This README keeps only the portfolio-safe technical summary.

