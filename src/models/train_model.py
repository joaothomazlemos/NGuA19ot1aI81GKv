#---------Importing libraries---------#

#---Data analysis---#
import pandas as pd
import numpy as np

#---evaluation---#
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

#---visualization---#
import matplotlib.pyplot as plt

#---utils---#
import os
import joblib
from pathlib import Path



#---------Models---------#
from xgboost import XGBClassifier


def split_data(df_train, df_test):
    # test data
    X_test = df_test.drop('y', axis=1).values
    y_test = df_test['y'].values

    # matrix and vector data undersampled thats used for training
    X_train = df_train.drop('y', axis=1).values
    y_train = df_train['y'].values

    data = {
        'train':{'X':X_train, 'y':y_train},
        'test':{'X':X_test, 'y':y_test}
    }
    return data



#train and return models

def train_model(data, xgboost_param):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    custom_scorer = make_scorer(f1_score)

    #grid search
    grid = RandomizedSearchCV(
        estimator=XGBClassifier(),
        param_distributions=xgboost_param,
        scoring=custom_scorer,
        cv=cv,
        n_iter=10,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    X_train = data['train']['X']
    y_train = data['train']['y']

    #fit model
    grid.fit(X_train, y_train)

    #return model
    return grid.best_estimator_



def model_metrics(model, data):
    preds = model.predict(data['test']['X'])
    f1 = f1_score(data['test']['y'], preds)

    metrics_dict = {
        'f1_score': f1
    }
    return metrics_dict


def main():
    print('Training model...')
    #---data---#



    #data folder
    

    data_path = Path('train_model.py').resolve().parent.parent / 'data' / 'preprocessed'
    print('data_path: ', data_path)
    test_path = data_path / 'df_test.csv'
    train_path = data_path / 'df_train.csv'
    
    print('test_path: ', test_path)
    print('train_path: ', train_path)

    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)

    # XGBoost parameters
    xgboost_params = {
    'n_estimators': [100, 200, 400, 800, 1600],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [2, 4, 8, 16, 32],
    'gamma': [0, 0.1, 0.5, 1.0],
    'subsample': [0.5, 0.75, 1.0],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss']
    }

    # split data
    data = split_data(df_train, df_test)

    # train model
    xgb = train_model(data, xgboost_params)

    # model evaluation
    model_metrics(xgb, data)

    # save model in the same folder

    model_name = 'xgboost.pkl'
    model_path = Path('train_model.py').resolve().parents[0] / model_name

    joblib.dump(xgb, model_path)


if __name__ == '__main__':
    main()
            

        

