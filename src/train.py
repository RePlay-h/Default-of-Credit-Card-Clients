import pandas as pd
import numpy as np

import pickle
import yaml
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import loguniform
from catboost import CatBoostClassifier
import optuna

def train_baseline(X_train, y_train, model_path):
   
    # Train baseline model
    lr = LogisticRegression(max_iter=300)
    distributions = {
        'C': loguniform(1e-2, 1e2),
        'penalty': ['l1', 'l2', 'elasticnet']
    }

    # Tuning baseline model
    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=distributions,
        n_iter=30,
        scoring='auc-roc',
        cv=7,
        random_state=101
    )

    random_search.fit(X_train, y_train)

    print(f'Best params (LogisticRegression): {random_search.best_params_}')
    print(f'Best score (LogisticRegression): {random_search.best_score_}')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(random_search, open(model_path, 'wb'))

    print(f'Save LogisticRegression baseline model into pickle-file')    


# special function to find the best hyperparams for CatBoost with optuna
def objective(trial):
    # define hyperparams for optuna
    params = {
        'iterations': trial.suggest_int("iterations", 500, 1000),
        'depth': trial.suggest_int("depth", 2, 3, 5, 7, 10),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-2, 0.5),
        'l2_leaf_reg': trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e2),
        'border_count': trial.suggest_int("border_count", 32, 128, 254),
        'random_strength': trial.suggest_float("random_strength", 0.0, 10.0),
        'eval_metric': "AUC",
        'verbose': 0,
        'random_state': 101,
    }

    model = CatBoostClassifier(**params)

    aucs = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc'
    )

    return aucs.mean()

def train(X_train, y_train, boosting_path):
    # define categorial features (SEX, EDUCATION, MARRIAGE)
    cat_features = [1, 2, 3]

    # select hyperparams
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=75)

    print(f"Best params (CatBoost): {study.best_params}")
    print(f"Best AUC on train (CatBoost): {study.best_value}")

    best_model = CatBoostClassifier(**study.best_params)
    best_model.fit(X_train, y_train, cat_feature=cat_features)

    os.makedirs(os.path.dirname(boosting_path), exist_ok=True)
    pickle.dump(best_model, open(boosting_path, 'wb'))

    print(f'Save CatBoost model into pickle-file')    




# save test samples into csv-files
def save_test_samples(X_test, y_test, X_test_path, y_test_path):
    X_test.to_csv(X_test_path)
    y_test.to_csv(y_test_path)

if __name__ == '__main__':
    # get dataset
    params = yaml.safe_load(open("params.yaml"))['train']
    dataset = pd.csv_read(params['input'])

    # One hot-encoding for EDUCATION feature
    df_encoded = pd.get_dummies(dataset, columns=['EDUCATION'], drop_first=True, dtype=np.int64)

    # Create train-test samples
    X, y = df_encoded.drop('default payment next month', axis=1), df_encoded['default payment next month']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    # train baseline
    train_baseline(X_train, y_train, params['baseline_path'])

    # train gradient boosting model
    train(X_train, y_train, params['boosting_path'])

    # save test split
    save_test_samples(X_test, y_test, params['output_X'], params['output_y'])
