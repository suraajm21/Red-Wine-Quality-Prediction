#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')


# ## Loading Dataset 

# In[2]:


df = pd.read_csv(r"C:\Users\User\Documents\Projects\ML Projects\Red Wine Quality Prediction\winequality-red.csv")
print("Successfully Imported Data!")
df


# ## Data Profiling 

# ### Shape of the dataset 

# In[3]:


print(df.shape)


# ## Description 

# ### Statistical summary of data 

# In[4]:


df.describe()


# ### Feature information 

# In[5]:


df.info()


# ## Exploratory Data Analysis

# ### Structure of the Dataset

# In[6]:


df_shape = df.shape
print("The credit rating dataset has", df_shape[0], "records, each with", df_shape[1],"attributes")


# ## Finding Null Values 

# In[7]:


print(df.isna().sum())


# In[8]:


df.corr()


# In[9]:


df.groupby('quality').mean()


# In[10]:


#Observation:

#All the features are numeric. No need of any encoding techniques


# # Data Analysis

# ## Countplot: 

# In[11]:


sns.countplot(df['quality'])
plt.show()


# In[12]:


sns.countplot(df['pH'])
plt.show()


# In[13]:


sns.countplot(df['alcohol'])
plt.show()


# In[14]:


sns.countplot(df['fixed acidity'])
plt.show()


# In[15]:


sns.countplot(df['volatile acidity'])
plt.show()


# In[16]:


sns.countplot(df['citric acid'])
plt.show()


# In[17]:


sns.countplot(df['density'])
plt.show()


# ## KDE plot: 

# In[18]:


sns.kdeplot(df.query('quality > 2').quality)


# ## Distplot: 

# In[19]:


sns.distplot(df['alcohol'])


# In[20]:


df.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[21]:


df.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# ## Histogram 

# In[22]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# ## Checking for Multicollinearity

# ## Heatmap for expressing correlation 

# In[23]:


plt.figure(figsize =(15,10))
sns.heatmap(df.corr(),robust=True,fmt='.1g',linewidths=1.3,linecolor='gold',annot=True);


# ## Pair Plot: 

# In[24]:


sns.pairplot(df)


# ## Violinplot: 

# In[25]:


sns.violinplot(x='quality', y='alcohol', data=df)


# ## Feature Selection

# ### Create Classification version of target variable 

# ####  Separate feature variables and target variable
# 

# In[26]:


df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
X = df.drop(['quality','goodquality'], axis = 1)
Y = df['goodquality']


# #### See proportion of good vs bad wines

# In[27]:


df['goodquality'].value_counts()


# In[28]:


X


# In[29]:


print(Y)


# ## Feature Importance 

# In[30]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# # Model selection and building

# ## Splitting training and testing data

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# ## LogisticRegression: 

# In[32]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[33]:


confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# ## K-Nearest Neighbours:

# In[34]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# ## Support Vector Classifier: 

# In[35]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))


# ## Decision Tree:

# In[36]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# ## GaussianNB:

# In[37]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# ## Random Forest Classifier:

# In[38]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# In[39]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','K-Nearest Neighbours', 'Support Vector Classifier','Decision Tree' ,'GaussianNB','Random Forest Classifier'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# In[40]:


#Observation:

#Random Forest Classifier performs well than other models

#Hence I will use Random Forest Classifier for training my model.


# In[41]:


conf_mat2=confusion_matrix(Y_test,y_pred2)
plt.figure(figsize=(6,6))
sns.heatmap(conf_mat2,annot=True,fmt=".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


# In[ ]:





# In[ ]:




