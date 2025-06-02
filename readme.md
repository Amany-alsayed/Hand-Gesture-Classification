
# Hand Gesture Classification – Model Training & Selection

This repository contains the research phase of the Hand Gesture Classification project. It includes model training, evaluation, and experiment tracking using **MLflow**.

---

## 🧪 Setup Instructions

 **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running MLflow and Experiments

1. **Start MLflow UI**

   ```bash
   mlflow ui
   ```

   This will serve the tracking UI at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

2. **Run Experiment Script**

   ```bash
   python main.py
   ```

   This script trains different models and logs metrics, parameters, and artifacts to MLflow.

---

## ✅ Model Selection

We trained and evaluated the following models:

- `SVMClassifier`
- `GradientBoostingClassifier`
- `RandomForestClassifier`

### 🔍 Why SVMClassifier Was Chosen for Deployment

The **SVMClassifier** was selected for deployment because it achieved the best overall performance on validation metrics, including accuracy, F1-score, precision, and recall. It also had the highest **cross-validation (CV) score**, which indicates generalization to unseen data.

---

## 📊 Model Comparison Table

| Model                   | Mean CV Score | Accuracy | F1-score | Precision | Recall  |
|------------------------|---------------|----------|----------|-----------|---------|
| SVMClassifier           | **0.9723**    | **0.9753** | **0.9753** | **0.9755**  | **0.9753** |
| GradientBoostingClassifier | 0.9528        | 0.9595   | 0.9596   | 0.9600    | 0.9595  |
| RandomForestClassifier  | 0.8443        | 0.7244   | 0.7243   | 0.7610    | 0.7244  |

## 📌 Notes

- All experiments are tracked using MLflow and can be viewed via the UI.
- The selected model (SVM) is used in the deployment_Hand-Gesture-Classification_project repository.
```
https://github.com/Amany-alsayed/deployment_Hand-Gesture-Classification_project
```

