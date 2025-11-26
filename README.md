# EE016-ML-digit-recognition

**Author:** Steve Li  
**Course:** EE016 - Introduction to Machine Learning  
**Institution:** University of California, Riverside  
**Term:** Fall 2025

## Project Overview

This project performs a comparative analysis of Convolutional Neural Networks (CNNs) versus traditional machine learning methods (SVM, Logistic Regression, KNN) for handwritten digit recognition. We evaluate model performance on two datasets to assess generalization and robustness.

## Datasets

| Dataset | Samples | Image Size | Description |
|---------|---------|------------|-------------|
| **MNIST** | 70,000 | 28×28 | Standard benchmark for digit recognition |
| **USPS** | 9,298 | 16×16 | Real-world postal service digits |

## Project Structure

```
├── phase2_data_processing.ipynb   # Data loading, preprocessing, visualization
├── phase3_model_training.ipynb    # Model development and hyperparameter tuning
├── phase4_evaluation.ipynb        # Final evaluation and PCA analysis
├── data/                          # Dataset files (not tracked)
├── outputs/                       # Generated figures and saved models
└── README.md
```

## Methods

### Traditional ML Methods
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Deep Learning
- Convolutional Neural Network (CNN) with varying architectures

### Dimensionality Reduction
- Principal Component Analysis (PCA)

## Installation

```bash
# Clone the repository
git clone https://github.com/Miracle197/EE016-ML-digit-recognition.git
cd mnist-usps-digit-recognition

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow h5py
```

## Usage

1. **Phase 2 - Data Processing:**
   ```bash
   jupyter notebook phase2_data_processing.ipynb
   ```

2. **Phase 3 - Model Training:**
   ```bash
   jupyter notebook phase3_model_training.ipynb
   ```

## Key Findings

*(To be updated after completing the project)*

- CNN achieves highest accuracy on MNIST (XX%)
- Cross-dataset evaluation on USPS shows...
- PCA reduces dimensionality while maintaining...

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- h5py


## License

This project is for educational purposes as part of UC Riverside coursework.
