# Machine Learning Projects

A collection of machine learning projects exploring regression, classification, and clustering techniques applied to real-world datasets.

---

## Projects

### 1. Student Performance Prediction
> Supervised Learning · Linear Regression

Predicts a student's final grade (`G3`) based on study habits and prior academic performance.

**Features used:** Study time, number of absences, first period grade (`G1`), second period grade (`G2`)

**Pipeline**
- Load and preprocess dataset (semicolon-delimited)
- Feature selection and train/test split
- Feature scaling with `StandardScaler`
- Model training with Linear Regression
- Evaluation using MAE and RMSE

**Metrics**
```
Mean Absolute Error   (MAE)  : X.XX
Root Mean Squared Error (RMSE) : X.XX
```

---

### 2. Fraud Detection
> Supervised Learning · Logistic Regression

Identifies fraudulent credit card transactions from a heavily imbalanced dataset.

**Dataset** — Credit Card Fraud Detection via Kaggle  
Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection  
Place `fraudTrain.csv` and `fraudTest.csv` in the project root before running.

**Pipeline**
- Load pre-split train and test sets
- Select numerical features and apply feature scaling
- Train Logistic Regression with `class_weight='balanced'`
- Evaluate with confusion matrix and classification report

**Metrics**
```
Confusion Matrix
Classification Report (Precision · Recall · F1-Score)
```

---

### 3. Customer Segmentation
> Unsupervised Learning · K-Means Clustering

Groups customers into distinct segments based on age, income, and spending behavior.

**Features used:** Age, Annual Income, Spending Score

**Pipeline**
- Load and explore customer dataset
- Feature scaling for distance-based clustering
- K-Means clustering to assign segment labels
- Export results to `clustered_customers.csv`

**Output**
```
Cluster distribution across all customers
Sample records with assigned cluster labels
Saved: clustered_customers.csv
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas | Data loading and manipulation |
| scikit-learn | Model training and evaluation |
| numpy | Numerical operations |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/username/ml-projects.git
cd ml-projects

# Install dependencies
pip install pandas scikit-learn numpy
```

---

## Usage

```bash
# Run individual projects
python student_performance/predict.py
python fraud_detection/detect.py
python customer_segmentation/segment.py
```

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

Please keep code clean and well-commented.

---

## License

This project is licensed under the [MIT License](LICENSE).
