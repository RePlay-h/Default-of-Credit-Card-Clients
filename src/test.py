
import pandas as pd
import numpy as np

from sklearn import metrics
import mlflow 
import mlflow.sklearn
from mlflow.models import infer_signature

import yaml
import json
import pickle
import os

def calculate_metrics(model_name, X_test, y_test, model_path):
    with mlflow.start_run(run_name=model_name):
        # open pickle-file with the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # calculate metrics
            # precision, recall, f1, auc
            precision = metrics.precision_score(y_test, y_pred, average='binary', labels=[1])
            recall = metrics.recall_score(y_test, y_pred, average='binary', labels=[1])
            f1 = metrics.f1_score(y_test, y_pred, average='binary', labels=[1])
            auc = metrics.roc_auc_score(y_test, y_proba)

            # print metrics
            print(model_name)
            print(f'        auc: {auc}')
            print(f'        precision: {precision}')
            print(f'        recall: {recall}')
            print(f'        f1: {f1}')

            signature = infer_signature(X_test, y_pred)
            input_example = X_test.iloc[:5]

            # log metrics with mlflow
            mlflow.log_metric('auc', auc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1', f1)

            mlflow.log_params(model.get_params())

            if model_name.lower().startswith('logistic'):
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="LogisticRegression",
                    signature=signature,
                    input_example=input_example
                 )
            else:
                mlflow.catboost.log_model(
                    cb_model=model, 
                    artifact_path="CatBoost",
                    signature=signature,
                    input_example=input_example
                    )

            # print confusion matrix
            print('Confusion matrix')
            print(metrics.confusion_matrix(y_test, y_pred))

            metrics_dict = {
                'model_name' : model_name,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
            return metrics_dict


if __name__ == '__main__':
    # load parameters for this file
    params = yaml.safe_load(open('params.yaml'))['test']

    # read test samples
    X_test = pd.read_csv(params['X_test'])
    y_test = pd.read_csv(params['y_test'])


    # calculate metrics for Logistic regression
    baseline_metrics = calculate_metrics(
        'LogisticRegression', 
        X_test, 
        y_test, 
        params['baseline_path']
        )

    # calculate metrics for CatBoost
    catboost_metrics = calculate_metrics(
        'CatBoost', 
        X_test, 
        y_test, 
        params['boosting_path']
        )
    
    my_metrics = [baseline_metrics, catboost_metrics]
    
    # save metrics
    os.makedirs(os.path.dirname(params['metrics_path']), exist_ok=True)

    with open(params['metrics_path'], 'w', encoding='utf-8') as f:
        json.dump(my_metrics, f, indent=4)