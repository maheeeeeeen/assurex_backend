# AssureX Claim Engine — 8-Model Training & Benchmark Comparison Report

**Dataset Scale:** 10,000 Unique Claims (7,000 Train / 1,500 Val / 1,500 Test)
**Winning Model:** **XGBoost** (Test Accuracy: **89.80%**)

## Performance Comparison Table (Test Set — 1,500 Records)

| Rank | Model Name | Test Accuracy | Weighted F1 | Precision | Recall | Manual Review F1 | 5-Fold CV F1 | Inference Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest** | **89.80%** | 0.9004 | 0.9209 | 0.8980 | 0.9179 | 0.8806 | 0.098 ms |
| 2 | **XGBoost** | **89.80%** | 0.9004 | 0.9214 | 0.8980 | 0.9177 | 0.8809 | 0.011 ms |
| 3 | **LightGBM** | **89.80%** | 0.9004 | 0.9214 | 0.8980 | 0.9177 | 0.8798 | 0.012 ms |
| 4 | **Decision Tree** | **89.60%** | 0.8985 | 0.9187 | 0.8960 | 0.9187 | 0.8798 | 0.002 ms |
| 5 | **SVM** | **82.87%** | 0.8294 | 0.8371 | 0.8287 | 0.8133 | 0.8024 | 1.549 ms |
| 6 | **Logistic Regression** | **76.93%** | 0.7632 | 0.8264 | 0.7693 | 0.6868 | 0.7396 | 0.001 ms |
| 7 | **KNN** | **75.73%** | 0.7580 | 0.7610 | 0.7573 | 0.7468 | 0.7638 | 4.109 ms |
| 8 | **Naive Bayes** | **69.47%** | 0.6960 | 0.8403 | 0.6947 | 0.6550 | 0.6827 | 0.012 ms |

## Hyperparameter Tuning Summary

- **Random Forest**: Best Params = `{"max_depth": 20, "min_samples_split": 2, "n_estimators": 100}` | CV F1 = 0.8806
- **XGBoost**: Best Params = `{"learning_rate": 0.03, "max_depth": 8, "n_estimators": 100, "subsample": 0.8}` | CV F1 = 0.8809
- **LightGBM**: Best Params = `{"learning_rate": 0.03, "max_depth": 6, "n_estimators": 100, "num_leaves": 15}` | CV F1 = 0.8798
- **Decision Tree**: Best Params = `{"criterion": "gini", "max_depth": 10, "min_samples_split": 2}` | CV F1 = 0.8798
- **SVM**: Best Params = `{"C": 10.0, "kernel": "rbf"}` | CV F1 = 0.8024
- **Logistic Regression**: Best Params = `{"C": 0.01, "solver": "lbfgs"}` | CV F1 = 0.7396
- **KNN**: Best Params = `{"metric": "manhattan", "n_neighbors": 11, "weights": "distance"}` | CV F1 = 0.7638
- **Naive Bayes**: Best Params = `{"var_smoothing": 0.001}` | CV F1 = 0.6827

## Per-Class Metric Breakdown (Test Set)

| Model | Likely Valid F1 | Likely Invalid F1 | Manual Review F1 | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8683 | 0.9152 | **0.9179** | 89.80% |
| **XGBoost** | 0.8685 | 0.9152 | **0.9177** | 89.80% |
| **LightGBM** | 0.8685 | 0.9152 | **0.9177** | 89.80% |
| **Decision Tree** | 0.8656 | 0.9113 | **0.9187** | 89.60% |
| **SVM** | 0.8168 | 0.8584 | **0.8133** | 82.87% |
| **Logistic Regression** | 0.7643 | 0.8385 | **0.6868** | 76.93% |
| **KNN** | 0.7426 | 0.7848 | **0.7468** | 75.73% |
| **Naive Bayes** | 0.6867 | 0.7462 | **0.6550** | 69.47% |
