# Developer Role Classification

This project implements a machine learning system to classify developers into different roles based on their skills, experience, and project history.

## Project Structure

- **notebooks/**: Jupyter notebooks for different stages of the project
  - `1_preprocessing_notebook.ipynb`: Data cleaning and feature engineering
  - `2_exploratory_analysis_notebook.ipynb`: Data visualization and exploration
  - `3_modeling_notebook.ipynb`: Model training and evaluation

- **src/**: Source code for the project
  - `preprocess.py`: Script for data preprocessing
  - `train.c`: C implementation of neural network model
  - `tensor.c` & `tensor.h`: Tensor operations for the neural network

- **data/**: Dataset files
  - `final_dataset.csv`: Cleaned and preprocessed dataset
  - `processed_dataset.csv`: Dataset after feature engineering

- **docs/**: Documentation files
  - `evaluation_report.md`: Comprehensive evaluation of model performance
  - `design_reflection.md`: Reflection on design decisions
  - `annotated_examples.md`: Annotated examples of classifications

## Key Results

- Primary metric: Macro F1 score of 0.978
- Model accuracy: 98.6%
- Strong per-class precision and recall across all developer roles

## Reproducibility

To ensure reproducibility, all random seeds have been fixed to 42 across all libraries:
- NumPy
- TensorFlow
- PyTorch
- Scikit-learn
- Random (Python standard library)

## Environment Setup

```bash
# Install required packages
pip install -r requirements.txt

# For the C implementation
gcc -o train train.c tensor.c -lm
```

## Running the Model

```bash
# Preprocess the data
python preprocess.py

# Train and evaluate the model
./train
```

## Evaluation Metrics

The model is evaluated using:
- Macro F1 score (primary metric)
- Per-class precision and recall
- Confusion matrix
- Calibration assessment
