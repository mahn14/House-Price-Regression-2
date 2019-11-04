## Dependencies
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 

# Read Raw CSV files
def read_raw(filepath):
    ''' Return Pandas DataFrames of raw training and testing sets  
    '''
    # Read raw csv
    train = pd.read_csv(filepath + 'train.csv').drop('Id', axis=1).reset_index(drop=True)
    test = pd.read_csv(filepath + 'test.csv').drop('Id', axis=1).reset_index(drop=True)
    return(train, test)

def get_lengths(filepath):
    train = pd.read_csv(filepath + 'train.csv').drop('Id', axis=1).reset_index(drop=True)
    test = pd.read_csv(filepath + 'test.csv').drop('Id', axis=1).reset_index(drop=True)
    return(train.shape[0], test.shape[0])    

def get_SalePrice(filepath):
    train, test = read_raw(filepath)
    return(train['SalePrice'])

####################################################################################################
## DROP FEATURES AND OBSERVATIONS
####################################################################################################


# Get columns based on number of missing obs
def get_cols(df, frac=0.1):
    ''' Returns columns to use for modeling and exploration
        Omits features missing more than 'frac' of their observations
    '''
    
    # Remove missing features
    missing = df.isnull().sum() / df.shape[0]
    cols = list(missing[missing <= frac].index)
    return(cols)


# Drop 2 outliers
def drop_outliers(df):
    ''' Returns Pandas DataFrame without 2 possible outliers
    '''

    outliers = np.sort(df['SalePrice'])[-2:]
    new_train = df[df['SalePrice'] < np.min(outliers)].reset_index(drop=True)
    
    return(new_train)


# Drop features with too many missing obs
def drop_missing(df, frac=0.1):
    ''' Returns DataFrame after dropping features with more than 'frac' missing values
    '''
    cols = get_cols(df, frac=frac)
    df = df[cols]
    return(df)


# Drop outliers and features
def drop_data(filepath, frac=0, outliers=False):
    ''' Return :: train, test
        Drop outliers and features with missing observations
    '''
    
    # Read Data
    train, test = read_raw(filepath)
    
    # Drop Outliers
    if outliers == True:
        train = drop_outliers(train)
    df_full = pd.concat([train.drop('SalePrice', axis=1, inplace=False), test], axis=0)
    
    # Drop Missing Features
    df = drop_missing(df_full, frac=frac)
    
    # Split Data
    length = train.shape[0]
    df_train = df.iloc[:length, :]
    df_train['SalePrice'] = train['SalePrice']
    
    df_test = df.iloc[length:, :]
    
    return(df_train, df_test)


####################################################################################################
## IMPUTATIONS
####################################################################################################


def get_missing_features(df):
    missing = df.isnull().sum()
    features = missing[missing > 0]
    return(features)

def get_missing_dataframe(df):
    features = get_missing_features(df)
    df_missing = pd.DataFrame(features, columns=['Missing']).sort_values('Missing')
    return(df_missing)

def impute_garage(df):
    # Single missing values
    df['GarageCars'].fillna(0, inplace=True)
    df['GarageArea'].fillna(0, inplace=True)
    
    # Many missing values
    features_garage2 = ['GarageType', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageQual']
    df[features_garage2] = df[features_garage2].fillna('None')
    
    return(df)

def impute_bsmt(df):
    features_num = ['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath']
    df[features_num] = df[features_num].fillna(0)
    
    features_cat = ['BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtExposure', 'BsmtCond']
    df[features_cat] = df[features_cat].fillna('None')
    return(df)


def impute_others(df):
    df['Electrical'] = df['Electrical'].fillna('None')
    df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
    df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
    df['Functional'] = df['Functional'].fillna('Typ')
    df['KitchenQual'] = df['KitchenQual'].fillna('TA')
    df['MSZoning'] = df['MSZoning'].fillna('RL')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['SaleType'] = df['SaleType'].fillna('WD')
    df['Utilities'] = df['Utilities'].fillna('AllPub')

    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    try:
        df['LotFrontage'] = df['LotFrontage'].fillna(60)
    except:
        pass 
    return(df)

    
def impute_df(df):
    # train and test imputes garage, basement the same
    df = impute_bsmt(df)
    df = impute_garage(df)
    df = impute_others(df)

    return(df)

####################################################################################################
## PREPROCESS
####################################################################################################


def preprocess(filepath, frac=0.2, outliers=False):
    # Get DataFrame
    train, test = drop_data(filepath, frac=frac, outliers=outliers)
    y = train['SalePrice']
    df_full = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
    df_full = impute_df(df_full)

    return(df_full, y)