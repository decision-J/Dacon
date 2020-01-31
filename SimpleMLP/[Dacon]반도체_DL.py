#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import time


# In[2]:


train = pd.read_csv('train.csv')

X = train.iloc[:,4:] 
X = X.iloc[:,0::2] #짝수번호만
y1 = train.loc[:, train.columns == 'layer_1']/10
y2 = train.loc[:, train.columns == 'layer_2']/10
y3 = train.loc[:, train.columns == 'layer_3']/10
y4 = train.loc[:, train.columns == 'layer_4']/10

test = pd.read_csv('test.csv')
test = test.iloc[:,1::2]

train_data = X
train_targets = train.iloc[:,:4]/10

test_data = test


# In[29]:


from keras import models
from keras import layers

import keras
keras.__version__


# In[54]:


def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    return model


# In[55]:


k = 3
num_val_samples = len(train_data) // k
num_epochs = 20
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비: k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 케라스 모델 구성(컴파일 포함)
    model = build_model()
    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


# In[38]:


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# In[39]:


import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[43]:


min(average_mae_history) # 11 epoch


# In[48]:


# 새롭게 컴파인된 모델을 얻습니다
model = build_model()
# 전체 데이터로 훈련시킵니다
model.fit(train_data, train_targets,
          epochs=11, batch_size=16, verbose=0)


# In[63]:


pred = model.predict(test_data)

pred = pd.DataFrame(pred*10)

pred.to_csv('pred_DL11.csv')


# In[ ]:




