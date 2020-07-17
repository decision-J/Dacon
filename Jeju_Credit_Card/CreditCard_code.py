import pandas as pd
import numpy as np
import os
import warnings

import seaborn as sns
warnings.filterwarnings(action='ignore')

os.getcwd()
os.chdir('C:\\Users\\JYW\\Desktop\\dataset')
# should change the directory

dataset = pd.read_csv('201901-202003.csv')
dataset.info()
dataset.head()
dataset.isnull().sum()
print(dataset[dataset.isnull().any(axis=1)].head())

submission = pd.read_csv('submission.csv')
submission.isnull().sum()
submission.info()
submission.head()
submission.CARD_SIDO_NM.value_counts()
submission.STD_CLSS_NM.value_counts()
set(dataset.STD_CLSS_NM)

len(set(dataset.STD_CLSS_NM))
len(submission.STD_CLSS_NM.value_counts())

len(set(dataset.CARD_SIDO_NM))
len(submission.CARD_SIDO_NM.value_counts())

import matplotlib.pyplot as plt
max(dataset.AMT/10000)

sns.distplot(dataset.AMT/10000)
plt.title("AMT distplot")
plt.show()

sns.distplot(dataset[dataset.AMT/10000 < 10000].AMT/10000)
plt.title("AMT distplot")
plt.show()

sns.distplot(dataset[dataset.AMT/10000 >= 10000].AMT/10000)
plt.title("AMT distplot")
plt.show()

sns.boxplot(x = "AMT",  data = dataset)
plt.show()

len(dataset[dataset.AMT/10000 < 1000])
len(dataset[dataset.AMT/10000 < 2000]) - len(dataset[dataset.AMT/10000 < 1000])
len(dataset[dataset.AMT/10000 < 10000]) - len(dataset[dataset.AMT/10000 < 2000])
len(dataset[dataset.AMT/10000 >= 10000])


## Baseline : RF regressor
from sklearn.ensemble import RandomForestRegressor
