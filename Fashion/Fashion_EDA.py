import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
os.chdir('C:/Users/JYW/Desktop/Github/repository/Dacon/landmark')
os.getcwd()

train_data = pd.read_csv('./train.csv')
category = pd.read_csv('./category.csv')
submission = pd.read_csv("./sample_submisstion.csv")

train_data["landmark_name"] = [train_data["id"][i][:-4] for i in range(len(train_data))]

print("Training data size",train_data.shape)
print("category data size",category.shape)
submission.head()

train_data['id'].value_counts().hist()
# No 중복

total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
# No missing

# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp

plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['landmark_id'])

plt.show()
