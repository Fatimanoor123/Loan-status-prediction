# 🏦 Loan Status Prediction using Random Forest Classifier

This project predicts whether a loan will be **approved (Y)** or **not approved (N)** using supervised machine learning based on applicant details like income, credit history, employment, and more.

---

## 📊 Problem Statement

Given historical loan application data, the goal is to **train a model** that can classify future loan applications into **Approved (1)** or **Not Approved (0)** categories.

---

## 📁 Dataset Description

- `train.csv`: Contains both **features and labels** (Loan_Status).
- `test.csv`: Contains only features; used for final prediction.

---

## 🛠️ Tools and Libraries Used

- Python
- Pandas & NumPy — Data handling
- Matplotlib & Seaborn — Visualization
- Scikit-learn — ML modeling and evaluation

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/loan-status-prediction.git
cd loan-status-prediction
```



## 2. 📦 Install Dependencies

Make sure you have Python 3.x installed.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 3. 📁 Add Dataset

Place `train.csv` and `test.csv` files in the **root project directory**.

📎 [Sample Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

---

## 4. ▶️ Run the Script

To execute the code, run:

```bash
python loan_prediction.py
```
## 🔍 Model Workflow

- 📝 **Load and Inspect Data**
- 🧹 **Handle Missing Values** using forward fill
- 🗃️ **Drop Unnecessary Columns** like `Loan_ID`
- 🔄 **Encode Labels** (Y/N → 1/0) and One-Hot Encode categorical features
- 🧩 **Align Train and Test Columns**
- ✂️ **Split Data** (80% training / 20% validation)
- 🌲 **Train Random Forest Model**
- 📋 **Evaluate using classification report**
- 🔮 **Predict on test set**

---

## 📊 Model Evaluation

After training, you’ll get a classification report including:

- ✅ **Accuracy**
- 📏 **Precision**
- 📈 **Recall**
- 📊 **F1-score**

---

## 📌 Project Structure

```bash
📁 loan-status-prediction/
│
├── loan_prediction.py       # Main script
├── train.csv                # Training dataset
├── test.csv                 # Test dataset
└── README.md                # This file



