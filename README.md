# MNIST Classification on Amazon SageMaker

A comprehensive implementation of MNIST digit classification using PyTorch on Amazon SageMaker, achieving 99.4% test accuracy with a parameter-efficient CNN architecture.

## 📊 Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9):
- Training set: 60,000 images
- Test set: 10,000 images
- Input dimensions: 1x28x28 (single channel grayscale)
- 10 output classes (digits 0-9)

## 🏗️ Model Architecture

The model uses a modern, efficient CNN architecture with the following key components:

### Feature Extraction Blocks
1. **Initial Block**
   - 3 consecutive Conv2D layers (1→10→10→10 channels)
   - Each followed by ReLU, BatchNorm, and Dropout(0.05)
   - MaxPool2D reduction

2. **Deep Feature Block**
   - 3 Conv2D layers (10→10→10 channels)
   - ReLU + BatchNorm + Dropout(0.05) after each
   - Focus on feature refinement

3. **Final Feature Block**
   - Single Conv2D layer (10→12 channels)
   - ReLU + BatchNorm + Dropout(0.05)

### Classification Head
- Global Average Pooling
- 1x1 Conv2D for final classification (12→10 channels)
- LogSoftmax activation

### Architecture Highlights
- Parameter efficient (<20K parameters)
- Heavy use of regularization (BatchNorm + Dropout)
- No fully connected layers
- Residual-style information flow

## 🚀 Training on SageMaker

### Training Configuration
- **Framework**: PyTorch on SageMaker
- **Instance Type**: ml.g4dn.xlarge
- **Epochs**: 20
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Negative Log Likelihood

![image](https://github.com/user-attachments/assets/8eb8870d-6381-4e6b-bbf0-603d620d9af7)


### Data Augmentation
1. **Random Rotation**
   - Range: -7° to +7°
   - Fill value: 1 (white)
   - Purpose: Improve rotation invariance

2. **CoarseDropout (Cutout)**
   - 1-3 holes per image
   - Hole size: 4-5 pixels
   - Purpose: Reduce overfitting

### Training Results

The model achieves excellent performance metrics:

- **Final Test Accuracy**: 99.4%
- **Final Training Accuracy**: 98.43%
- **Final Test Loss**: 0.0187
- **Final Training Loss**: 0.0511

#### Training Progression
- Quick initial convergence (92.6% → 96.6% in first 5 epochs)
- Steady improvement in middle epochs
- Stable final performance with minimal overfitting
- Test accuracy consistently higher than training accuracy, indicating good generalization

![7th assignment](https://github.com/user-attachments/assets/9b08bd87-78f1-47e1-bb18-e6979476bec5)

## 📈 Performance Analysis

### Loss Curves
- Training loss: Smooth descent from 0.247 to 0.051
- Test loss: Stable decrease from 0.061 to 0.019
- No signs of overfitting (test loss remains below training loss)

### Accuracy Curves
- Training accuracy: Steady climb from 92.6% to 98.43%
- Test accuracy: Strong performance from early epochs, reaching 99.4%
- Consistent ~1% gap between train and test accuracy

## 🛠️ Implementation Details

### Project Structure
```
mnist_classifier/
├── model_3.py          # Model architecture definition
├── train.py            # Training script with SageMaker integration
├── mnist_inference.py  # Inference and evaluation script
└── README.md          # Project documentation
```

### Key Features
- Efficient CNN architecture
- Comprehensive data augmentation
- SageMaker integration
- Detailed metrics tracking
- Misclassification analysis capability

## 🔍 Model Evaluation

The model shows excellent performance characteristics:
- High accuracy (99.4% test)
- Stable training progression
- Good generalization (test > train accuracy)
- Efficient parameter usage
- Robust to input variations (via augmentation)

## 🚀 Future Improvements

Potential areas for enhancement:
1. Experiment with additional augmentation techniques
2. Implement learning rate scheduling
3. Explore model pruning for further efficiency
4. Add model interpretability analysis
5. Implement ensemble methods for even higher accuracy

