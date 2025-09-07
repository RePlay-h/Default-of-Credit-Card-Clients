import pandas as pd
import numpy as np

import pickle
import yaml
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
from catboost import CatBoostClassifier
import optuna

def train_baseline(df, model_path):
   
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

    print(f'Best params: {random_search.best_params_}')
    print(f'Best score: {random_search.best_score_}')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(random_search, open(model_path, 'wb'))

    print(f'Save LogisticRegression baseline model into pickle-file')    

    
    
def train(df, X_train, y_train, boosting_path):
    pass


def save_test_samples(X_test, y_test):
    pass

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
    train_baseline(dataset, X_train, y_train, params['baseline_path'])

    # train gradient boosting model
    train(dataset, X_train, y_train, params['boosting_path'])

    # save test split
    save_test_samples(X_test, y_test, params['output_X'], params['output_y'])
