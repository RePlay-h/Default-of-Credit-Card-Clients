
import pandas as pd

from sklearn import metrics

import yaml
import pickle
import os

def calculate_metrics(model_name, X_test, y_test, model_path, metrics_save):
    with open(model_path, 'rb') as f:
        baseline_model = pickle.load(f)
        y_pred = baseline_model.predict(X_test)
        
        # calculate metrics
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        # precision, recall, f1
        precision = metrics.precision_score(y_test, y_pred, average='binary')
        recall = metrics.recall_score(y_test, y_pred, average='binary')
        f1 = metrics.f1_score(y_test, y_pred, average='binary')

        print(model_name)
        print(f'        precision: {precision}')
        print(f'        recall: {recall}')
        print(f'        f1: {f1}')

        os.makedirs(os.path.dirname(metrics_save), exist_ok=True)

        with open(metrics_save, 'w') as f:
            f.write(model_name)
            f.write(f'precision: {precision}')
            f.write(f'recall: {recall}')
            f.write(f'f1: {f1}')

if __name__ == '__main__':
    # load parameters for this file
    params = yaml.safe_load(open('params.yaml'))['test']

    # read test samples
    X_test = pd.read_csv(params['X_test'])
    y_test = pd.read_csv(params['y_test'])

    # calculate metrics for Logistic regression
    calculate_metrics(
        'LogisticRegression', 
        X_test, 
        y_test, 
        params['baseline_path'], 
        params['baseline_metrics']
        )

    # calculate metrics for CatBoost
    calculate_metrics(
        'CatBoost', 
        X_test, 
        y_test, 
        params['boosting_path'], 
        params['boosting_metrics']
        )
