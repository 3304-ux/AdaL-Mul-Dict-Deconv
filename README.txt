## 1. Overview
The model is designed to address the lateral spatial non-stationarity of seismic wavelets via an adaptive learned iterative shrinkage-thresholding algorithm (AdaLISTA).

## 2. File Description
All source code files are independent and runnable (no compressed packages):

- DATA.py
  - Defines the custom seismic dataset class (`SeismicDataset`) for loading and organizing training data
  - Builds PyTorch DataLoader with a customized collation function
  - Configures computing device (GPU/CPU) automatically

- Model_supervised.py
  - Implements the core AdaLISTA network (the main deep learning model)
  - Includes multi-dictionary selection and forward propagation logic
  - Supports adaptive parameter constraints and gradient stabilization

- Supervised_learning_training.py
  - Main training and testing pipeline for the multi-dictionary model
  - Implements MSE loss + L1 regularization loss function
  - Includes Adam optimizer, learning rate scheduling, and model evaluation
  - Provides the `main()` function to start model training

## 3. Environment Requirements
The code is compatible with the following environment:
- Python 3.10
- PyTorch 2.0
- NumPy
- SciPy (for loading .mat data)
- Matplotlib (optional, for visualization)
