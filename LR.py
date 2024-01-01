#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv("dataset.csv")

df


# In[2]:


from sklearn.linear_model import LinearRegression

# 'trestbps'를 예측하는 선형 회귀 모델 생성
lr_regression = LinearRegression()

# 회귀에 사용할 데이터 준비 ('age'와 'oldpeak'을 포함)
X_regression = df[['age', 'oldpeak']]
y_regression = df['trestbps']

# 선형 회귀 모델 학습
lr_regression.fit(X_regression, y_regression)

# 학습된 회귀 모델을 통해 예측 수행
y_pred_regression = lr_regression.predict(X_regression)


# In[4]:


from sklearn.metrics import mean_squared_error, r2_score

# MSE 계산
mse = mean_squared_error(y_regression, y_pred_regression)
print(f"Mean Squared Error (MSE): {mse}")

# R-squared 계산
r2 = r2_score(y_regression, y_pred_regression)
print(f"R-squared: {r2}")


# In[ ]:




