#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt


# In[2]:


pd.read_csv("dataset.csv")
 #age : 연령 <범주형, 명목형>
 #sex : 성별 <범주형, 명목형> 
 #cp : 가슴통증유형(0 ~ 4) 
 #trestbps : 안정혈압 <연속형 데이터>
 #chol : 혈청 코레스테롤 <연속형 데이터>
 #fbs : 공복 혈당(1=true, 2=false) 
 #restecg : 안전 심전도 결과(0,1,2)
 #thalach : 최대 심박동수 <명목형>
 #exang : 협심증 유발 운동(1 = yes, 0= no) 
 #oldpeak : 비교적 안정되기까지 운동으로 유발되는 ST depression
 #slope : 최대 운동 ST segement의 기울기
 #ca : 형광 투시된 주요 혈관의 수(0,1,2,3)
 #thal : 3=보통 6= 해결된 결함 7 = 해결가능한 결함
 #target : 심장병 진단 (1=true, 0=false) 


# In[4]:


df=pd.read_csv("dataset.csv")
missing_values = df.isnull().sum()
print(missing_values)


# In[5]:


# 각 열에 결측치가 없다는 것을 확인할 수 있다. 고로 전처리 과정에서 결측치를 다루는 과정을 필요가 없다. 

#이상치 유무 조회 
 #이상치 : 일반적인 데이터 패턴에서 크게 벗어나는 값으로, 대부분 데이터들과는 매우 다른 값을 의미한다. 이상치는 데이터 수집과정에서 
    #발생할 수 있으며, 실제로 드문 이벤드일 수도 있다.
    # 극단치 : 일반적인 데이터 패턴에서 벗어나는 개별적인 데이터 포인트이다. 이상치는 대부분 데이터들과 매우 다른 값을 갖기 때문에
    #주로 특이한 사례나 오류로 간주된다.
    
    #특이치 : 이상치와 유사하지만, 특이치는 전체적인 데이터 패턴에서 벗어난 전체적은 그룹이나 패턴을 가질 수 있다. 이는 에를 들어 특정 이벤트의 
    #발생이나 특별한 조건에서는 특이한 패턴일 수 있다.
    


# 데이터 불러오기 (예: df는 당신이 가지고 있는 데이터프레임)
# df = pd.read_csv('your_dataset.csv')
import matplotlib.pyplot as plt
# Boxplot으로 이상치 확인
plt.figure(figsize=(10, 6))  # 여기서 괄호 안에 =(10, 6)을 넣어줘야 합니다.
df.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Dataset')
plt.show()


# In[7]:


#범주형 데이터
 #범주형 데이터는 일정한 범주 또는 카테고리로 나뉘어진 데이터를 의미한다. 이는 일반적으로 명목형 데이터와 순서형 데이터로 나뉜다.
    
#1. 명목형 데이터 : 범주 간에 순서가 없는 데이터이다. 예컨대, 성별, 혈액형, 도시이름 등은 명목형 데이터이다. 
#이러한 데이터는 분류에 사용되며, 각 범주 사이에는 순서나 계층 구조가 없다.

#2. 순서형 데이터 : 범주 간에 순서가 있는 데이터이다. 예컨대, 교육수준이나 만족도는 순서형 데이터의 예시이다. 

# 주어진 데이터는 모든 열들이 수치화가 되어 있는 것으로 보인다. 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report




# In[12]:


from sklearn.svm import SVC, SVR

# 분류 모델 (심장병 발병 유무를 예측하는 SVM)
svm_classifier = SVC(kernel='linear')  # 선형 SVM 분류기
svm_regression = SVR(kernel='linear')  # 선형 SVM 회귀 모델


# In[13]:


# 분류에 사용할 데이터 준비
X_classification = df.drop('target', axis=1)
y_classification = df['target']


# In[14]:


# 회귀에 사용할 데이터 준비
X_regression = df.drop('trestbps', axis=1)  # trestbps를 제외한 나머지 특성들
y_regression = df['trestbps']


# In[15]:


# 데이터 분할 (학습용과 테스트용)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# SVM 모델 학습


# In[16]:


# SVM 모델 학습
svm_classifier.fit(X_train_clf, y_train_clf)
svm_regression.fit(X_train_reg, y_train_reg)


# In[17]:


# 분류 모델 평가
classification_accuracy = svm_classifier.score(X_test_clf, y_test_clf)
print(f"Classification Accuracy: {classification_accuracy}")



# In[18]:


# 회귀 모델 평가
regression_score = svm_regression.score(X_test_reg, y_test_reg)
print(f"Regression Score: {regression_score}")


# In[19]:


correlation_matrix = df.corr()
trestbps_correlation = correlation_matrix['trestbps'].sort_values(ascending=False)
print(trestbps_correlation)


# In[20]:


# 'age'와 'oldpeak'을 포함하여 회귀에 사용할 데이터 준비
X_regression = df[['age', 'oldpeak']]
y_regression = df['trestbps']


# In[21]:


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)


# In[22]:


svm_regression.fit(X_train_reg, y_train_reg)


# In[23]:


# 회귀 모델 평가
regression_score = svm_regression.score(X_test_reg, y_test_reg)
print(f"Regression Score: {regression_score}")


# In[ ]:




