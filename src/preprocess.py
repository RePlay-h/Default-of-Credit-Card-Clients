import pandas as pd
from sklearn.preprocessing import LabelEncoder

import yaml


def preprocess(dataset):
    pass

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['preprocess']
    
    df = pd.read_excel(params['input'])

    df.to_csv(params['output'])

    print(f'Processed dataset: {params['output']}')

    

