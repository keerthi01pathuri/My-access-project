#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("./CO2 Emissions_Canada.csv")
df.head(2)


# In[3]:


df.shape


# In[4]:


df.info


# In[5]:


df.describe


# In[6]:


df.isna().sum()


# In[7]:


df = df.dropna()


# In[8]:


df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.shape


# In[11]:


num_cols = df.iloc[:,1:-1].select_dtypes(exclude='object').columns.values
num_cols


# In[12]:


le = LabelEncoder()


# In[13]:


for i in df.iloc[:,1:-1].columns:
    if df[i].dtype == 'object':  
        df[i] = le.fit_transform(df[i])


# In[14]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[15]:


df.head(2)


# In[16]:


X = df.iloc[:,1:-1]
y = df.iloc[:,-1]


# In[17]:


y.value_counts()


# In[18]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


sc = StandardScaler()
X_train[num_cols] = sc.fit_transform(X_train[num_cols])
X_test[num_cols] = sc.fit_transform(X_test[num_cols])


# In[21]:


X_test


# In[22]:


# Create a Linear Regression model and fit it
lr =LinearRegression()
model = lr.fit(X_train,y_train)


# In[23]:


y_predict=model.predict(X_test)
y_predict


# In[24]:


from sklearn.metrics import accuracy_score
r2_score(y_predict,y_test)


# In[25]:


DT=DecisionTreeRegressor()
dt=DT.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
dt_acc_score=r2_score(y_test,dt_pred)*100
(dt_acc_score)


# In[26]:


RF=RandomForestRegressor()
rf=RF.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
rf_acc_score=r2_score(y_test,rf_pred)*100
(rf_acc_score)


# In[ ]:




