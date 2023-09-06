#!/usr/bin/env python
# coding: utf-8

# # Task 1 : Prediction using Supervised ML
# # Predicting the percentage of student based on no of study hours    

# In[38]:


# Importing required libraries

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


# In[2]:


# Loading the dataset

df = pd.read_excel(r"C:\Users\vinsl\OneDrive\student_details.xlsx")


# In[3]:


df.head(5)


# In[40]:


df.isnull().sum()


# In[41]:


df.duplicated().sum()


# In[4]:


df.info()


# In[20]:


df.corr()


# In[21]:


sn.heatmap(df.corr())

## It shows Marks obtained and Hours spend are highly correlated
# In[18]:


pt.grid()
pt.scatter(x = df['Hours'],y = df['Scores'])
pt.ylabel("Score of the student")
pt.xlabel("Hours spend")
pt.title("Student's percentage based on hours studied")
pt.show()


# In[19]:


pt.grid()
pt.scatter(x = df['Hours'],y = df['Scores'])
sn.regplot(x = df['Hours'],y = df['Scores'])
pt.ylabel("Score of the student")
pt.xlabel("Hours spend")
pt.title("Regression plot")
pt.show()


# # Training the model

# In[23]:


# Splitting the data

x = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30)

#Applying linear regression to the model

ln = linear_model.LinearRegression()
model = ln.fit(x_train,y_train)


# # Predicting the marks

# In[30]:


#predicting the marks of the students based on hours

predict = model.predict(x_test)
predicted_marks = pd.DataFrame({
    'Hours' : [i for i in x_test],
    'Predicted Hours' : [j for j in predict]
})
predicted_marks


# # Comparing the results

# In[32]:


#comparing original marks with predicted marks
actual_vs_predict = pd.DataFrame({
    'Actual_marks' : y_test,
    'Predicted_marks' : predict
})
actual_vs_predict


# In[37]:


#Lets compare our result visually
pt.grid()
pt.title("Comapring actual marks with predicted marks")
pt.xlabel("Hours spend")
pt.ylabel("Marks obtained")
pt.scatter(x = x_test, y = y_test, color='blue')
pt.plot(x_test,predict, color='red')
pt.show()


# # Measuring the accuracy 

# In[39]:


#Measuring the accuracy of our prediction using Mean_absolute_error(measures error in prediction)

print("Mean Absolute Error : ",mean_absolute_error(y_test,predict)) 

# Closer the value of MAE, more accurate the model is. Here MAE is 4.29, which is not bad.
# # What will be the predicted score if the student studies for 9.25 hrs/days?

# In[52]:


Hours = [9.25]
result = model.predict([Hours])
print("Score obtained for studying 9.25hrs/day : {}".format(result))


# # Model shows 87.35% will obtain, if student studies 9.25hrs/day

# In[ ]:




