#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


# In[39]:


dataset=pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Priyanka\Datasets\winequality-red.csv")


# In[40]:


dataset  ## Quality of wine depends on its composition


# In[41]:


## visualization, checking which factor influences on the quality and how
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


sns.barplot(x="quality",y="fixed acidity",data=dataset)


# In[43]:


sns.barplot(x="quality",y="volatile acidity",data=dataset)  ##volatile acidity for good quality wine is less


# In[44]:


sns.barplot(x="quality",y="chlorides",data=dataset)  ##chlorides for good quality wine is less


# In[45]:


sns.barplot(x="quality",y="total sulfur dioxide",data=dataset)  ##sulphur for good quality wine is less


# In[46]:


sns.barplot(x="quality",y="alcohol",data=dataset) ##Alcohol increases as quality of wine increases


# In[47]:


dataset["quality"].unique()    ##quality is given interms of ratings


# In[48]:


dataset.info()


# In[49]:


dataset["quality"]=dataset["quality"].astype(float)


# In[50]:


##dividing the quality in good and bad depending on the rating
bins=(2,6,10)
names= ["bad","good"]
dataset["quality"]= pd.cut(dataset["quality"],bins=bins,labels=names)


# In[51]:


dataset["quality"].unique()


# In[52]:


##label encoding 0=bad and 1=good
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
dataset["quality"]= encoder.fit_transform(dataset["quality"])
dataset["quality"].unique()


# In[53]:


##defining x data and y data
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[54]:


##checking y counts 
print(Counter(y))


# In[55]:


##balancing the y data
from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_data,y_data=smote.fit_resample(x,y)


# In[56]:


print(Counter(y_data))


# In[57]:


##train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.25,random_state=111)


# In[58]:


## model fitting using ensemble technique


# In[71]:


from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1.fit(x_train,y_train)
y_pred=l1.predict(x_test)
acc=accuracy_score(y_pred,y_test)
#print("predicted values",y_pred)
print("accuracy score",acc.round(4)*100,"%")


# In[72]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
acc=accuracy_score(y_pred,y_test)
#print("predicted values",y_pred)
print("accuracy score",acc.round(4)*100,"%")


# In[77]:


from sklearn.svm import SVC
s=SVC(random_state=111)
s.fit(x_train,y_train)
y_pred=s.predict(x_test)
acc=accuracy_score(y_pred,y_test)
#print("predicted values",y_pred)
print("accuracy score",acc.round(4)*100,"%")


# In[74]:


##ensemble technique baggibg classifier 
from sklearn.ensemble import BaggingClassifier
bgg=BaggingClassifier(base_estimator=s,n_estimators=6,random_state=21)
bgg.fit(x_train,y_train)
y_pred=bgg.predict(x_test)
acc=accuracy_score(y_pred,y_test)
#print("predicted values",y_pred)
print("accuracy score",acc.round(4)*100,"%")


# In[63]:


## lets try to increase the efficiency by crossvalidation technique


# In[75]:



from sklearn.model_selection import KFold
kfold=KFold(6)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
score=cross_val_score(vote,x_data,y_data,cv=kfold)
print(np.mean(score))


# In[65]:


# Voting classifier :ensemble 


# In[81]:


from sklearn.ensemble import VotingClassifier
vote=VotingClassifier(estimators=[("logistics",l1),("bagging",bgg),("knn",knn),("svm",s)])
vote.fit(x_train,y_train)
y_pred=vote.predict(x_test)
acc=accuracy_score(y_pred,y_test)
#print("predicted values",y_pred)
print("accuracy score",acc.round(4)*100,"%")


# In[ ]:


#conclusion :
From the above model selection we say that ,the before using the cross validation the moodels gives the more than 80% accuracy,and 
the cross validation and bagging classifier gives the 70% and 76% score for the accurate prediction ,but it shows that this two techniques 
give the less accuracy than model ,due to that we again implement the another techniques :Ensemble Techniques ,it gives the 80% accuracy 
and from thei we say that ,this technique gives the proper prediction .
Hence we choose the Ensemble techiniques.

