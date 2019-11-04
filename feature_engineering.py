import preprocessing as PRE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import scipy.stats as stats
from scipy.special import boxcox1p

import warnings
warnings.filterwarnings('ignore')

def encode(df):
    df = df.copy()
    
    # Numerical to String
    col_to_str = ['MSSubClass', 'OverallQual', 'YrSold', 'MoSold']
    
    # Categorical to LabelEncode
    col_to_encode = ['BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'BsmtFinType1', 
            'BsmtFinType2', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold', 'KitchenQual', 'Functional', 'OverallQual']
    
    # Apply
    df[col_to_str] = df[col_to_str].applymap(str)
    df[col_to_encode] = df[col_to_encode].apply(LabelEncoder().fit_transform)

    return(df)


def split_num_cat(df):
    df = df.copy()
    
    # Split by object datatype
    num_features = df.dtypes[df.dtypes != "object"].index
    cat_features = df.dtypes[df.dtypes == "object"].index
    
    df_num = df[num_features]
    df_cat = df[cat_features]
    
    return(df_num, df_cat)

def get_boxcox(df, threshold=0.75, l=0.15):
    df = df.copy()
    
    # Check Skew
    skewness = df.apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
    skewness = skewness[abs(skewness) > threshold]

    skewed_features = skewness.index
    l = l
    for f in skewed_features:
        df[f] = boxcox1p(df[f], l)
        
    return(df)

def get_dummies(df):
    df = df.copy()
    df = pd.get_dummies(df)
    return(df)

def normalize(x):
    x = np.array(x)
    x = (x - x.min()) / (x.max() - x.min())
    return(x)

def feature_engineer(df, boxcox=True, threshold=0.75, l=0.15, scale=True):
    df = df.copy()
    
    df_encoded = encode(df)
    df_num, df_cat = split_num_cat(df_encoded)
    if boxcox:
        df_num = get_boxcox(df_num, threshold=threshold, l=l)
    df_cat = get_dummies(df_cat)
    
    
    df_full = pd.concat([df_num, df_cat], axis=1)
    
    if scale:
        df_full = df_full.apply(normalize)
   
    
    return(df_full)