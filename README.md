# Fashion MNIST Classification with PyTorch

## Project Overview

This project implements and compares three neural network architectures for classifying images from the Fashion MNIST dataset using PyTorch. The goal is to evaluate the performance of different model complexities in terms of accuracy, loss, and training time.

## Dataset

- **Name**: Fashion MNIST
- **Source**: torchvision.datasets.FashionMNIST
- **Size**: 60,000 training images, 10,000 test images
- **Image Dimensions**: 28x28 pixels, grayscale
- **Classes**: 10 categories of clothing items

## Model Architectures

### 1. FashionMnistModelV0 (Simple Linear Model)
```python
class FashionMnistModelV0(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=output_shape)
    )
```

### 2. FashionMnistModelV1 (Multi-Layer Perceptron)
```python
class FashionMnistModelV1(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_shape),
        nn.ReLU()
    )
```

### 3. FashionMNISTModelV2 (Convolutional Neural Network)
```python
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
```

## Training Configuration

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: SGD (Learning Rate: 0.1)
- **Batch Size**: 32
- **Epochs**: 3
- **Device**: CUDA (if available), else CPU

## Results

| Model               | Test Accuracy | Test Loss | Training Time (s) |
|---------------------|---------------|-----------|-------------------|
| FashionMnistModelV0 | 83.43%        | 0.4766    | 29.97             |
| FashionMnistModelV1 | 75.02%        | 0.6850    | 32.10             |
| FashionMNISTModelV2 | 88.06%        | 0.3239    | 37.13             |

## Dependencies

- torch==1.x.x
- torchvision==0.x.x
- matplotlib==3.x.x
- numpy==1.x.x
- tqdm==4.x.x

## Future Work

- Implement data augmentation techniques
- Experiment with different optimizers (e.g., Adam, RMSprop)
- Increase the number of epochs for potentially better performance
- Implement early stopping to prevent overfitting
- Explore transfer learning with pre-trained models

