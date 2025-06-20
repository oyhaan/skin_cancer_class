````markdown
# Skin Cancer Classification Using Machine Learning

**Project Name:** Skin Cancer Classification Using Machine Learning

## Problem Statement
Late skin cancer diagnosis in Africa leads to high mortality rates due to limited diagnostic access, high treatment costs, and socio-cultural barriers. In 2020, skin cancer accounted for approximately 10,000 deaths annually in Africa, with over 90% of cases diagnosed at advanced stages (GLOBOCAN 2020). Existing screening efforts are constrained by funding, reach, and stigma, underscoring the need for an accessible, low-cost, and accurate tool for early detection in underserved regions.

## Dataset
The dataset for this project is sourced from the **ISIC (International Skin Imaging Collaboration) Archive**, a publicly available collection of dermoscopic and clinical images of skin lesions:

- **Source:** https://www.isic-archive.com/
- **Content:** Labeled images of malignant (e.g., melanoma) and benign lesions, plus metadata such as diagnosis and patient demographics.
- **Challenges:**
  - **Class Imbalance:** Fewer malignant samples require techniques to avoid model bias.
  - **Skin Tone Diversity:** Under-representation of darker skin tones demands careful preprocessing to maintain fairness.

## Implementation Overview
This project compares classical machine learning algorithms against neural network models for binary classification (malignant vs. benign).

1. **Classical ML Baseline (XGBoost)**
   - **Feature Extraction:** Images flattened into 1D vectors (172×251×3 pixels) and normalized.
   - **Hyperparameter Tuning:** GridSearchCV over `learning_rate`, `max_depth`, `n_estimators`, and `subsample`.
   - **Evaluation Metrics:** Accuracy, F1-score, Precision, Recall, ROC AUC.

2. **Neural Network Models**
   - **Baseline (Instance 1):** Simple dense network (64-unit hidden layer) with default settings.
   - **Optimized Instances (2–5):** Variations of optimizer (Adam, RMSprop), regularization (L1, L2), dropout, learning rate, early stopping, and layer count.
   - **Modularity:** Functions `define_model()`, `loss_curve_plot()`, `print_final_accuracy()`, and data loaders enable reusable code.

## Results and Discussion
### Neural Network Training Experiments
Below is the summary of five neural network training instances evaluated on the test set:

| Training Instance            | Optimizer | Learning Rate | Dropout | Regularizer | Early Stopping | Layers | Test Accuracy | F1-score | Precision | Recall |
|------------------------------|-----------|---------------|---------|-------------|----------------|--------|---------------|----------|-----------|--------|
| **1. Simple Baseline**       | Default   | 0.001         | 0.0     | None        | No             | 2      | 0.68          | 0.81     | 0.68      | 1.00   |
| **2. Adam + L2**             | Adam      | 0.001         | 0.3     | L2          | No             | 2      | 0.66          | 0.79     | 0.68      | 0.94   |
| **3. RMSprop + L1 + Dropout**| RMSprop   | 0.001         | 0.2     | L1          | No             | 2      | 0.68          | 0.81     | 0.68      | 1.00   |
| **4. Adam + Early Stopping** | Adam      | 0.005         | 0.4     | L2          | Yes            | 2      | 0.40          | 0.46     | 0.60      | 0.37   |
| **5. Combined Optimized**    | Adam      | 0.0001        | 0.5     | L2 (0.01)   | Yes            | 3      | 0.72          | 0.84     | 0.75      | 0.92   |

**Key Findings:**
- **Best Neural Configuration:** Instance 5 (low learning rate, dropout 0.5, L2 regularization, early stopping) achieved the highest balance of accuracy (0.72) and robustness (F1 0.84).
- **Hyperparameter Effects:** Increasing dropout and adding L2 reduced overfitting; a lower learning rate improved convergence; early stopping prevented wasted epochs.
- **Optimizer Comparison:** Adam and RMSprop yielded similar results, but Adam demonstrated more stable convergence in deeper configurations.

### Final Model Comparison
| Model         | Accuracy | F1-score | Precision | Recall | ROC AUC |
|---------------|----------|----------|-----------|--------|---------|
| Tuned XGBoost | 0.98     | 0.98     | 0.98      | 0.98   | 0.9946  |

**Conclusion:**
- **Best Overall Model:** Tuned XGBoost outperformed all neural network variants on tabular features by a wide margin, highlighting its strength on structured data.
- **Neural vs. Classical:** While optimized neural nets improved over the baseline, XGBoost’s built-in regularization and tree-based learning captured image features more effectively when flattened.

## Running the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/oyhaan/skin_cancer_class.git
   cd skin_cancer_class
````

2. Install dependencies:

   ```bash
   pip install tensorflow scikit-learn numpy pandas seaborn matplotlib xgboost pillow
   ```
3. Open and run `notebook.ipynb` in Jupyter or Colab.

   * If using Colab, mount Google Drive at `/content/drive/MyDrive/ISIC-images/`.
   * Ensure `saved_models/` directory exists for loading and saving models.

## Loading the Best Model

```python
import tensorflow as tf
# Load the optimized CNN (Instance 5)
best_cnn = tf.keras.models.load_model('saved_models/optimized_cnn_model_4.h5')
best_cnn.summary()

# Load the optimized XGBoost
import joblib
best_xgb = joblib.load('saved_models/optimized_xgb_model.pkl')
```

## Repository Contents

* `notebook.ipynb` — project code and analysis (modularized).
* `README.md` — this overview and instructions.
* `saved_models/` — contains 6 CNN models (`model1.h5`–`model5.h5`) and the optimized XGBoost model.
* Video Presentation: [https://www.loom.com/share/e6e8729657d74da5be7af81ba8b692b0?sid=e169e634-2daf-4637-a387-e9a1c2aa1445](https://www.loom.com/share/e6e8729657d74da5be7af81ba8b692b0?sid=e169e634-2daf-4637-a387-e9a1c2aa1445)
* Dataset Source: [https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D)

```
```
