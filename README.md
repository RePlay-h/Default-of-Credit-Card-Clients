# Default of Credit Card Clients

This project focuses on predicting the probability of default for credit card holders based on customer data from Taiwanese banks.  
The dataset is taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).  

Predicting defaults is important for banks as it helps reduce credit risk, adjust credit limits, and optimize lending decisions.  

---

## 📊 Dataset

The raw data contains information on **30,000 clients**, including:  
- Credit limit (`LIMIT_BAL`)  
- Gender, education, marital status, age  
- Payment history over 6 months (`PAY_0 … PAY_6`)  
- Bill amounts (`BILL_AMT1 … BILL_AMT6`)  
- Payment amounts (`PAY_AMT1 … PAY_AMT6`)  
- Target variable `default_payment_next_month` — whether the client defaulted in the next month  

The classes are highly imbalanced (about 22% defaults), so handling class imbalance was taken into account during model training.  

---

## 🛠️ Feature Engineering

To improve model performance, new features were added:

1. **Number of Delinquencies**  
   Sum of indicators (PAY_i > 0) across all months  

2. **Debt Ratio**  
   Ratio of total debt to credit limit  

3. **Payment Ratio**  
   Ratio of debt to actual payment made  

---

## 🤖 Models and Metrics

- **Logistic Regression** — baseline model  
- **CatBoostClassifier** — main model, hyperparameters tuned with Optuna (TPE algorithm)  

The main metric is **AUC ROC**, due to class imbalance. Additionally, precision, recall, and F1-score were measured.  

### Logistic Regression
- AUC: `0.6396`  
- Precision: `0.0`  
- Recall: `0.0`  
- F1: `0.0`  

Confusion matrix:
```
[[7056 2]
[1942 0]]
```

### CatBoost
- AUC: `0.7801`  
- Precision: `0.6877`  
- Recall: `0.3584`  
- F1: `0.4712`  

Confusion matrix:
```
[[6742 316]
[1246 696]]
```

📌 CatBoost significantly outperformed Logistic Regression and showed a more balanced precision-recall performance.  

---

## 📂 Структура проекта
```
├── .dvc/ # служебные файлы DVC
├── catboost_info/ # служебные файлы CatBoost
├── data/ # данные
│ ├── raw/ # исходные данные
│ │ └── clients.xls
│ ├── processed/ # подготовленные данные
│ │ └── data.csv
│ └── test/ # отложенные тестовые данные
│ ├── X_test.csv
│ └── y_test.csv
├── env/ # окружение
├── metrics/ # метрики моделей
│ └── metrics.json
├── model/ # сохранённые модели
│ ├── baseline.pkl # Logistic Regression
│ └── boosting.pkl # CatBoost
├── src/ # исходный код
│ ├── preprocess.py # препроцессинг данных
│ ├── train.py # обучение моделей
│ └── test.py # инференс и проверка моделей
├── EDA.ipynb # разведочный анализ данных
├── params.yaml # параметры обучения
├── dvc.yaml # пайплайн DVC
├── dvc.lock # lock-файл DVC
├── requirements.txt # зависимости
└── README.md # описание проекта
```
## ✅ Conclusions

- Logistic Regression struggled with class imbalance and almost never predicted the positive class  
- CatBoost achieved a significant improvement (AUC ≈ 0.78) and better balanced precision and recall  
- Future improvements could include:  
  - using **PR AUC** as the main metric  
  - handling class imbalance with methods like SMOTE, undersampling, or scale_pos_weight
  