import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings(action='ignore')

os.getcwd()
os.chdir('C:\\Users\\JYW\\Desktop\\dataset')
# should change the directory

dataset = pd.read_csv('201901-202003.csv')
dataset.info()
dataset.head()
dataset.isnull().sum()
print(dataset[dataset.isnull().any(axis=1)].head())
