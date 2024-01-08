#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt


# In[2]:


pd.read_csv("winequality-red.csv")


# In[3]:


pd.read_csv("winequality-white.csv")


# In[4]:


RED = pd.read_csv("winequality-red.csv")
WHITE = pd.read_csv("winequality-white.csv") #레드아인 데이터와 화이트 와인 데이트를 각 RED와 WHITE 변수가 할당 


# In[5]:


missing_values = RED.isnull().sum()
print(missing_values)


# In[7]:


RED = pd.read_csv("winequality-red.csv", delimiter=';')

# 데이터 확인
print(RED.head())  # 데이터의 처음 몇 개 행 출력하여 확인


# In[8]:


WHITE = pd.read_csv("winequality-white.csv", delimiter=';')

print(WHITE.head())


# In[11]:


# 데이터 불러오기 (예: df는 당신이 가지고 있는 데이터프레임)
# df = pd.read_csv('your_dataset.csv')
import matplotlib.pyplot as plt
# Boxplot으로 이상치 확인
plt.figure(figsize=(10, 6))  # 여기서 괄호 안에 =(10, 6)을 넣어줘야 합니다.
RED.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Dataset')
plt.show()


# In[12]:


# 데이터 불러오기 (예: df는 당신이 가지고 있는 데이터프레임)
# df = pd.read_csv('your_dataset.csv')
import matplotlib.pyplot as plt
# Boxplot으로 이상치 확인
plt.figure(figsize=(10, 6))  # 여기서 괄호 안에 =(10, 6)을 넣어줘야 합니다.
WHITE.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Dataset')
plt.show()


# In[13]:


#분류 문제는 quality를 판단하는 것이고, 회귀는 residual sugar를 예측하는 것
#이전 과제와 마찬가지로 SVM, LR, DT, RT, KNN 모델을 사용


# In[14]:


#SVM 분류
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[15]:


from sklearn.svm import SVC, SVR

# 분류 모델 (심장병 발병 유무를 예측하는 SVM)
svm_classifier = SVC(kernel='linear')  # 선형 SVM 분류기
svm_regression = SVR(kernel='linear')  # 선형 SVM 회귀 모델


# In[16]:


# RED 데이터셋을 예측 모델에 사용하기 위해 입력 데이터와 타겟 데이터로 분할
X_red = RED.drop(columns=['quality'])  # 'quality' 열을 제외한 나머지 열들은 입력 데이터
y_red = RED['quality']  # 'quality' 열은 타겟 데이터

# 데이터 분할 (학습용 데이터와 테스트용 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.2, random_state=42)

# SVM 분류 모델 생성 및 학습
svm_classifier = SVC(kernel='linear')  # 선형 SVM 분류기 생성
svm_classifier.fit(X_train, y_train)  # 모델 학습

# 테스트 데이터에 대한 예측
y_pred = svm_classifier.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 분류 보고서 출력
print(classification_report(y_test, y_pred))


# In[18]:


from sklearn.metrics import mean_squared_error, r2_score


# 'residual sugar'를 예측하기 위해 다른 특성 선택 (예시로 'fixed acidity', 'volatile acidity', 'citric acid' 선택)
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['residual sugar']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 회귀 모델 생성 및 학습
svm_regressor = SVR(kernel='linear')  # 선형 SVM 회귀 모델
svm_regressor.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = svm_regressor.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R-squared)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 'quality'를 분류하기 위해 다른 특성 선택
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['quality']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 분류 모델 생성 및 학습
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # K값은 5로 설정
knn_classifier.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = knn_classifier.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도
classification_rep = classification_report(y_test, y_pred)  # 분류 보고서

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기
data = pd.read_csv("winequality-white.csv", delimiter=';')

# 'residual sugar'를 예측하기 위해 다른 특성 선택 (예시로 'fixed acidity', 'volatile acidity', 'citric acid' 선택)
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['residual sugar']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 회귀 모델 생성 및 학습
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # K값은 5로 설정
knn_regressor.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = knn_regressor.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R-squared)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


# In[21]:


from sklearn.ensemble import RandomForestRegressor
# 'residual sugar'를 예측하기 위해 다른 특성 선택 (예시로 'fixed acidity', 'volatile acidity', 'citric acid' 선택)
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['residual sugar']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RT 회귀 모델 생성 및 학습
rt_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # 트리 개수는 100으로 설정
rt_regressor.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = rt_regressor.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R-squared)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 'quality'를 분류하기 위해 다른 특성 선택
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['quality']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RT 분류 모델 생성 및 학습
rt_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # 트리 개수는 100으로 설정
rt_classifier.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = rt_classifier.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도
classification_rep = classification_report(y_test, y_pred)  # 분류 보고서

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[23]:


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
# 'residual sugar'를 예측하기 위해 다른 특성 선택 (예시로 'fixed acidity', 'volatile acidity', 'citric acid' 선택)
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['residual sugar']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DT 회귀 모델 생성 및 학습
dt_regressor = DecisionTreeRegressor(random_state=42)  # 기본 설정으로 모델 생성
dt_regressor.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = dt_regressor.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R-squared)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 'quality'를 분류하기 위해 다른 특성 선택
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['quality']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DT 분류 모델 생성 및 학습
dt_classifier = DecisionTreeClassifier(random_state=42)  # 기본 설정으로 모델 생성
dt_classifier.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = dt_classifier.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도
classification_rep = classification_report(y_test, y_pred)  # 분류 보고서

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 'quality'를 분류하기 위해 다른 특성 선택
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['quality']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression 모델 생성 및 학습
lr_classifier = LogisticRegression(max_iter=1000)  # 모델 생성 (max_iter는 반복 횟수)
lr_classifier.fit(X_train, y_train)  # 학습

# 테스트 데이터에 대한 예측
y_pred = lr_classifier.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도
classification_rep = classification_report(y_test, y_pred)  # 분류 보고서

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 'residual sugar'를 예측하기 위해 다른 특성 선택 (예시로 'fixed acidity', 'volatile acidity', 'citric acid' 선택)
X = data[['fixed acidity', 'volatile acidity', 'citric acid']]  # 입력 특성
y = data['residual sugar']  # 예측할 타겟

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LR 회귀 모델 생성 및 학습
lr_regressor = LinearRegression()  # 모델 생성
lr_regressor.fit(X_train, y_train)  # 학습

# 테스트 데이터에 대한 예측
y_pred = lr_regressor.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R-squared)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


# In[ ]:




