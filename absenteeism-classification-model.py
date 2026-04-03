#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Absenteeism_preprocessed.csv')


# In[3]:


data.head()


# In[4]:


data['Absenteeism Time in Hours'].median()


# In[5]:


target=np.where(data['Absenteeism Time in Hours']>3,1,0)
target


# In[6]:


data['Excessive Absenteeism']=target
data.head()


# In[7]:


target.sum()/target.shape[0]


# In[8]:


data_target=data.drop(['Absenteeism Time in Hours'],axis=1)


# In[9]:


data_target=data_target.drop(['Date'],axis=1)


# In[10]:


data_target.head()


# In[11]:


data_target.shape


# In[12]:


data_target.iloc[:,:14]


# In[13]:


data_target.iloc[:,:-1]


# In[14]:


unscaled=data_target.iloc[:,:-1]


# In[15]:


from sklearn.preprocessing import StandardScaler

absenteeism_scaler=StandardScaler()


# In[16]:


absenteeism_scaler.fit(unscaled)


# In[17]:


scaled_inputs=absenteeism_scaler.transform(unscaled)


# In[18]:


scaled_inputs


# In[19]:


scaled_inputs.shape


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


train_test_split(scaled_inputs,target)


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(scaled_inputs,target,train_size=0.8,shuffle=True,random_state=20)


# In[23]:


print(x_train.shape,y_train.shape)


# In[24]:


print(x_test.shape,y_test.shape)


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[26]:


mod=LogisticRegression()


# In[27]:


mod.fit(x_train,y_train)


# In[28]:


mod.score(x_train,y_train)


# In[29]:


##get intercept and coef for tableau use


# In[30]:


mod.intercept_


# In[31]:


mod.coef_


# In[32]:


features=unscaled.columns.values


# In[33]:


summary=pd.DataFrame (columns=['Feature'],data=features)
summary['Coefficient']=np.transpose(mod.coef_)
summary


# In[34]:


summary.index=summary.index+1


# In[35]:


summary.loc[0]=['Intecept',mod.intercept_[0]]
summary=summary.sort_index()
summary


# In[36]:


#weights/intercepts


# In[37]:


summary['Odd_ratio']=np.exp(summary.Coefficient)
summary


# In[38]:


summary.sort_values('Odd_ratio',ascending=False)


# In[40]:


mod.score(x_test,y_test)


# In[41]:


predict_proba=mod.predict_proba(x_test)
predict_proba


# In[42]:


predict_proba.shape


# In[43]:


predict_proba[:,1]


# In[ ]:


##save model


# In[44]:


import pickle


# In[47]:


with open('model','wb') as file:
    pickle.dump(mod,file)


# In[46]:


with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)


# In[ ]:




