import preprocessing as PRE
import feature_engineering as FE
import numpy as np
import pandas as pd

def get_data(filepath='Data/Raw/', frac=0.2, outliers=False, boxcox=True, threshold=0.75, l=0.15, log=True, scale=True):
    df_pre, y = PRE.preprocess(filepath=filepath, frac=frac, outliers=outliers)
    df_eng = FE.feature_engineer(df_pre, boxcox=boxcox, threshold=threshold, l=l, scale=scale)

    if log:
        y = np.log(y)
    df_load = pd.concat([df_eng.iloc[:len(y),:], y], axis=1)
    
    return(df_load)


def split_train_val(df, test_frac=0.25):
    df = df.copy().sample(frac=1).reset_index(drop=True)
    
    n_test = int(np.round(df.shape[0] * test_frac))

    df_train = df.iloc[n_test:,:]
    df_test = df.iloc[:n_test, :]
    
    X_train = df_train.drop('SalePrice', axis=1)
    y_train = df_train['SalePrice']
    
    X_test = df_test.drop('SalePrice', axis=1)
    y_test = df_test['SalePrice']
    
    return(X_train, y_train, X_test, y_test)

def load_data(filepath='Data/Raw/', frac=0.2, outliers=False, boxcox=True, threshold=0.75, l=0.15, log=True, test_frac=0.25, scale=True):

    df_pre = get_data(filepath=filepath, frac=frac, outliers=outliers, boxcox=boxcox, threshold=threshold, log=log, scale=scale)
    X_train, y_train, X_test, y_test = split_train_val(df_pre, test_frac=test_frac)
    return(X_train, y_train, X_test, y_test)