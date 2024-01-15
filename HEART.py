#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("heart.csv") #index_col=0: 특정 열을 인덱스 열로 사용하도록함으로 사둉에 정의, 사용해야하는 열이 실제 데이터 셋에는 제외될 수 있음 
df


# In[29]:


df


# In[30]:


# 컬렴명 확인
df.columns


# In[31]:


sns.countplot(data=df, x='target')
plt.xlabel('Disease or Not')
plt.ylabel('Count')
plt.title('Label')


# In[32]:


df['target'].value_counts()


# In[33]:


#결측치 확인
df.isnull().sum()


# In[34]:


df=df.dropna()


# In[35]:


#결측치 제거 후 다시 확인
df.isnull().sum()


# In[36]:


# 컬럼별 히스토그램 확인
# 'mean radius'와'mean texture' 칼럼을 선택

figure = plt.figure(figsize=(18,6))


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

# 'age'와 'thalach' 열을 선택하여 산점도 그리기
sns.scatterplot(data=df, x='age', y='thalach')

# 그래프 축 및 제목 설정
plt.xlabel('Age')
plt.ylabel('Max Heart Rate (thalach)')
plt.title('Scatter Plot of Age vs. Max Heart Rate')

# 그래프 표시
plt.show()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

# 데이터프레임에서 수치형 열을 선택하여 pairplot 그리기
sns.pairplot(df, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

# 그래프 표시
plt.show()




# In[39]:


# 각 컬럼별 히스토 그램 확인

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
plt.show()


# In[43]:


df


# In[46]:


X=df.drop('target',axis=1)


# In[47]:


X.head()


# In[48]:


df


# In[49]:


y=df['target']
y.value_counts()


# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[52]:


from sklearn.svm import SVC

clf_svm = SVC(random_state=0)
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print("\n--- SVM Classifier ---")
print(accuracy_score(y_test, pred_svm))
print(confusion_matrix(y_test, pred_svm))


# In[53]:


# Build a logistic regression classifier and predict

clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test)

print ("\n--- Logistic Regression Classifier ---")
print (accuracy_score(y_test, pred_lr))
print (confusion_matrix(y_test, pred_lr))


# In[54]:


clf_nn = MLPClassifier(random_state=0)
clf_nn.fit(X_train, y_train)

pred_nn = clf_nn.predict(X_test)

print ("\n--- Neural Network Classifier ---")
print (accuracy_score(y_test, pred_nn))
print (confusion_matrix(y_test, pred_nn))


# In[55]:


# Build a decision tree classifier and predict

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print ("\n--- Decision Tree Classifier ---")
print (accuracy_score(y_test, pred_dt))
print (confusion_matrix(y_test, pred_dt))


# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print ("\n--- Radom Forest ---")
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
print(accuracy_score(y_test,pred))
print (confusion_matrix(y_test, pred))


# In[ ]:




