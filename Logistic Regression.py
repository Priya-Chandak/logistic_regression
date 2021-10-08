#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[84]:


filename = 'titanic_data'
path = 'E:/desktop/ML/Logistic/{}.csv'.format(filename)
titanic_data = pd.read_csv(path)
titanic_data.head()


# In[85]:


## Analyse Data


# In[86]:


sns.countplot(x='Survived',data=titanic_data)


# In[87]:


sns.countplot(x='Survived',hue='Sex',data=titanic_data)


# In[88]:


sns.countplot(x='Survived',hue='Pclass',data=titanic_data)


# In[89]:


titanic_data['Age'].plot.hist()


# In[90]:


titanic_data['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[91]:


titanic_data.info()


# In[92]:


#Age of child in titanic


# In[93]:


sns.countplot(x='SibSp',data=titanic_data)


# In[94]:


# data Wrangling : removing Nan values or unnecessary values 


# In[95]:


titanic_data.isnull().sum()


# In[96]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# In[97]:


sns.boxplot(x='Pclass',y='Age',data=titanic_data)


# In[98]:


titanic_data.isnull()


# In[99]:


# titanic_data.dropna(inplace=True)
titanic_data.head(5)


# In[100]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# In[101]:


titanic_data.drop(['Cabin','Lifeboat','Body'],axis=1,inplace=True)
titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()


# In[102]:


titanic_data.head(2)


# In[103]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)


# In[104]:


embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark


# In[105]:


pc1=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
pc1.head(5)


# In[106]:


titanic_data=pd.concat([titanic_data,sex,embark,pc1],axis=1)
titanic_data.head(5)


# In[107]:


titanic_data.drop(['Class','Boarded','Age_wiki','WikiId','Sex','Embarked','Pclass','PassengerId','Name','Ticket','Name_wiki','Hometown','Destination'],axis=1,inplace=True)
titanic_data.head(5)


# ### Train dataset

# In[108]:


X = titanic_data.drop('Survived',axis=1)
y = titanic_data['Survived']


# In[116]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[131]:


logmodel = LogisticRegression(solver='lbfgs')


# In[132]:


logmodel.fit(X_train,y_train)


# In[134]:


predictions=logmodel.predict(X_test)


# In[137]:


classification_report(y_test,predictions)


# In[139]:


confusion_matrix(y_test,predictions)


# In[141]:


accuracy_score(y_test,predictions)


# ### Accuracy of the fitted model is 77%
