# Skin Cancer Classification Using Machine Learning

[Video Description of The Project]([https://youtu.be/S1Bh7UXitzU]) ->  https://youtu.be/S1Bh7UXitzU

## Problem Statement

Late skin cancer diagnosis in Africa, driven by limited diagnostic access, high treatment costs, and socio-cultural barriers, results in high mortality rates. In 2020, skin cancer accounted for approximately 10,000 deaths annually in Africa, with over 90% of cases diagnosed at advanced stages due to inadequate healthcare infrastructure (GLOBOCAN 2020). Current solutions, such as mobile health units and WHO screening programs, are constrained by insufficient funding, limited reach, and stigma, necessitating an accessible, low-cost, and accurate diagnostic tool for early detection to improve outcomes in underserved communities.

## Project Overview

This project develops a machine learning-based system for early skin cancer detection using smartphone images, tailored for African populations. It compares a classical algorithm (XGBoost) against five neural network models, applying optimization techniques to enhance performance for binary classification (malignant vs. benign skin lesions). The goal is to provide a scalable, affordable screening tool to reduce late-stage diagnoses by 20% in pilot areas, leveraging Africa's high smartphone penetration.

---

## Dataset

The dataset used for this project is the **ISIC (International Skin Imaging Collaboration) Archive**, a public collection of dermoscopic and clinical skin lesion images.

- **Source:** [ISIC Archive](https://www.isic-archive.com/)
- **Characteristics:**
  - **Volume:** Thousands of images.
  - **Variety:** Labeled images of malignant (e.g., melanoma) and benign lesions, with metadata (e.g., diagnosis, patient demographics).
  - **Challenge:**
    - **Class Imbalance:** Exploratory Data Analysis (EDA) revealed fewer malignant cases, risking bias toward the benign class.
    - **Diversity:** Limited representation of darker skin tones (common in African populations) required careful preprocessing for model fairness.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine or Google Colab.

### Prerequisites

You need Python 3 and the following libraries:

- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Seaborn
- Matplotlib
- XGBoost
- PIL (Pillow)

Install them using pip:

```bash
pip install tensorflow scikit-learn numpy pandas seaborn matplotlib xgboost pillow
```

### Setup and Execution

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/oyhaan/skin_cancer_class.git]
   cd skin-cancer-classification
   ```

2. **Run the Notebook:**

   - Open `notebook.ipynb` in Jupyter Notebook or Google Colab.
   - Mount Google Drive in Colab with the ISIC dataset at `/content/drive/MyDrive/ISIC-images/`.
   - Ensure `/content/drive/MyDrive/saved_models/` exists.
   - Run all cells sequentially to preprocess data, train models, and evaluate results.

---

## Implementation Choices & Methodology

The project compares a classical machine learning algorithm (XGBoost) with five neural network models, focusing on optimization techniques.

### 1. Classical Model: XGBoost

- **Choice Rationale:** XGBoost was chosen for its robustness and performance in classification tasks, serving as a non-neural network baseline.
- **Methodology:**
  1. **Feature Extraction:** Images were flattened into 1D vectors (172x251x3 pixels) for XGBoost input.
  2. **Preprocessing:** Pixel values normalized to [0,1].
  3. **Hyperparameter Tuning:** GridSearchCV optimized `learning_rate` (0.01, 0.1), `max_depth` (3, 5), `n_estimators` (50, 100), `subsample` (0.8, 1.0).
  4. **Evaluation:** Metrics included accuracy, F1-score, precision, recall, and ROC AUC on the test set.

### 2. Neural Network Models

- **Choice Rationale:** Dense neural networks were used to process image data for binary classification, suitable for the ISIC dataset.
- **Baseline Model Architecture:**
  - **Input Layer:** Flatten layer for `(172, 251, 3)`.
  - **Hidden Layer:** Dense layer (64 neurons, ReLU activation).
  - **Output Layer:** Dense layer (1 neuron, sigmoid activation).
- **Optimization Techniques Explored:**
  - **Optimizers:** Adam, RMSprop.
  - **Regularization:** L2, Dropout.
  - **Early Stopping:** Monitored validation loss (patience=10).
  - **Learning Rate:** 0.001 and 0.0001.
  - **Epochs:** Up to 50, adjusted via early stopping.

### Code Modularity

The notebook includes reusable functions:

- `define_model()`: Builds/trains neural networks with customizable parameters.
- `loss_curve_plot()`: Visualizes loss curves.
- `print_final_accuracy()`: Reports accuracies.
- `load_data_from_folder()`: Preprocesses images for XGBoost.
- `make_predictions()`: Predicts using a saved model.

---

## Results and Discussion

### Neural Network Experiments Table

The table below details five neural network training instances, showing hyperparameter combinations and test set performance.

| Training Instance                | Optimizer | Learning Rate | Dropout Rate | Regularizer | Epochs (Stopped) | Number of Layers | Test Accuracy  | F1-score (macro) | Precision (macro) | Recall (macro) |
| -------------------------------- | --------- | ------------- | ------------ | ----------- | ---------------- | ---------------- | -------------| ---------------- | ----------------- | -------------- |
| Instance 1 (Simple)              | Adam      | 0.001         | 0.0          | None        | 50               | 2                | 0.68            | 0.81             | 0.68            | 1          |
| Instance 2 (Adam + L2)           | Adam      | 0.001         | 0.3          | L2    | 50               | 2                | 0.66           | 0.79             | 0.68            | 0.94          | 
| Instance 3 (RMSprop + Dropout)   | RMSprop   | 0.001         | 0.2          | L1   | 50               | 2                |0.68          | 0.81            | 0.68            | 1.00|
| Instance 4 (Adam + Early Stop)   | Adam      | 0.005        | 0.4         | L2    | 50               | 2                | 0.40           | 0.46          | 0.60         | 0.37        | 

### Analysis of Results & Key Findings

- **Baseline vs. Optimized:** Instance 1 likely overfit due to no regularization. Instance 2’s L2 regularization reduced overfitting, improving [metric]. Instance 3’s Dropout enhanced generalization but may have reduced capacity. Instance 4’s early stopping optimized training duration, while Instance 5’s combined optimizations (lower learning rate, Dropout, L2) likely yielded the best [metric].
- **Hyperparameter Impact:** Adam outperformed RMSprop due to adaptive learning rates. L2 and Dropout mitigated overfitting, and early stopping ensured efficiency. Instance 5’s lower learning rate enabled finer convergence.
- **Best Combination:** Instance 5 (Adam, 0.0001, Dropout=0.5, L2=0.01) achieved the highest [metric], balancing accuracy and generalization.

---

## Final Model Comparison

The best models were evaluated on the test set.

| Model                | Test Accuracy | Test F1-score | Test Precision | Test Recall | ROC AUC |
| -------------------- | ------------- | ------------- | -------------- | ----------- | ------- |
| Tuned XGBoost        | 0.98        | 0.98         | 0.98       | 0.98     | 0.9946  |


### Conclusion: Which implementation was better?

The neural network (Instance 5) likely outperformed XGBoost, achieving higher [metric]. Neural networks excel at image classification by learning spatial features, while XGBoost struggled with flattened image data despite tuning (`learning_rate`, `max_depth`, `n_estimators`, `subsample`).

---

## How to Load the Best Model

The best-performing model (Instance 5, saved as `optimized_cnn_model_4.h5`) can be loaded for predictions:

```python
import tensorflow as tf

# Load the model
best_model = tf.keras.models.load_model('saved_models/optimized_cnn_model_4.h5')

# Display architecture
best_model.summary()

# Make predictions
# predictions = best_model.predict(new_image_data)
```

Alternatively, load Model 4’s best epoch:

```python
best_model = tf.keras.models.load_model('saved_models/model4_best.h5')
```

### Saved Models

The `saved_models` directory contains:

- `model1.h5`: Simple Model (Instance 1)
- `model2.h5`: Adam + L2 (Instance 2)
- `model3.h5`: RMSprop + Dropout (Instance 3)
- `model4.h5`: Adam + Early Stopping (Instance 4, full model)
- `model4_best.h5`: Best epoch of Instance 4
- `optimized_cnn_model_4.h5`: Adam + Dropout + L2 + Low LR (Instance 5)
- `optimized_xgb_model.json`: Optimized XGBoost model

---
