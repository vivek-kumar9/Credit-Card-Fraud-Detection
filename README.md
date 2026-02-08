# Credit Card Fraud Detection using Machine Learning

## 📌 Overview
This project focuses on detecting fraudulent credit card transactions in a highly imbalanced dataset using machine learning techniques. The goal is not just high accuracy, but effective fraud detection while minimizing false positives in a real-world alert system.

## 📂 Dataset
- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Transactions:** 284,807  
- **Fraud Rate:** ~0.17%  
- **Features:** PCA-transformed features (`V1–V28`) + `Amount`  
- **Target:** `Class` (1 = Fraud, 0 = Legitimate)

The dataset is not included in this repository due to size and licensing constraints.

## ⚙️ Tech Stack
- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

## 🔄 Project Workflow
1. Data loading and inspection  
2. Train–test split with stratification  
3. Handling class imbalance using SMOTE  
4. Baseline modeling with Random Forest  
5. Main modeling using XGBoost  
6. Model evaluation using ROC-AUC  
7. Precision–Recall analysis  
8. Threshold tuning to reduce false positives  
9. Feature importance analysis  

## 📊 Results
- **ROC-AUC:** ~0.95  
- **Key Outcome:** ~32% reduction in false positives via threshold tuning  
- **Focus:** Alert optimization rather than raw accuracy  

## 💡 Key Learnings
- Accuracy is misleading for highly imbalanced datasets  
- Threshold selection is a business decision, not a model default  
- XGBoost scales better than Random Forest for large imbalanced data  

## ⚠️ Limitations
- No temporal validation  
- PCA features limit interpretability  
- SMOTE may introduce synthetic bias  


## 🚀 Future Improvements
- Cost-sensitive learning  
- Time-based validation  
- Concept drift handling  
