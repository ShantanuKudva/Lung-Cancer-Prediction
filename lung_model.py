import numpy as np
import pandas as pd
import pickle

# **DATA ANALYSIS**


# **PRE PROCESSING**

from sklearn.preprocessing import FunctionTransformer # Transforming of Data
from sklearn.neighbors import KNeighborsRegressor

#Data collection and processing

#Loading the csv data to a pandas Dataframe
lung_data=pd.read_csv("data.csv")

#Dropping the index, patient id column
lung_data.drop(["Patient Id" , "index"], axis = 1 , inplace = True)

#Replacing low, mediun, high to 0,1,2 respectively
lung_data.replace(to_replace = "Low" , value = 0 , inplace = True)
lung_data.replace(to_replace = "Medium" , value = 1 , inplace = True)
lung_data.replace(to_replace = "High" , value = 2 , inplace = True)

a = lung_data.drop("Level" , axis = 1)
b = lung_data["Level"]

#checking the skewness of the data and seperating the right skewed data and left skewed data
right_skew = []
left_skew = []
for i in lung_data.columns:
    if lung_data[i].skew() > 0:
        right_skew.append(i)
    else:
        left_skew.append(i)


#for right skewed data, we square the terms and for the left skewed data we apply log transform
right_trf = FunctionTransformer(func = np.square)
left_trf = FunctionTransformer(func = np.log1p)
right_trfd = right_trf.fit_transform(lung_data[right_skew])
left_trfd = left_trf.fit_transform(lung_data[left_skew])

data_proc = pd.concat([right_trfd , left_trfd , b] , axis = 1 , join = "inner")

#splitting the data into testing and training data
train , test = np.split(lung_data.sample(frac = 1) , [int(0.8 * len(lung_data))])


def pre(dataframe):
    target = ["Level"]
    x = dataframe.drop(target, axis=1)
    y = dataframe[target]

    return x, y

X_train, Y_train = pre(train)
X_test, Y_test = pre(test)

model=KNeighborsRegressor()
model.fit(X_train, Y_train)
model.fit(X_test, Y_test)

#We will be using KNN regressor
#Building a predictive system



input_data=(73,1,5,6,6,5,6,5,6,5,8,5,5,5,4,3,6,2,1,2,1,6,2)


#change the data to a numpy array
input_data_as_numpy_array=np.array(input_data, dtype=float)

#reshape the numpy array as we are prediction for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

if(prediction[0]==0):
  print("The person does not have Lung Cancer")
elif(prediction[0]==1):
  print("The person could have lower levels of Lung Cancer")
else:
  print("The person could have higher levels of lung cancer")

filename="lung_model.sav"
pickle.dump(model, open(filename, "wb"))


input_data=(73,1,5,6,6,5,6,5,6,5,8,5,5,5,4,3,6,2,1,2,1,6,2)

#change the data to a numpy array
input_data_as_numpy_array=np.array(input_data, dtype=float)

#reshape the numpy array as we are prediction for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

loaded_model=pickle.load(open(filename, "rb"))
prediction=loaded_model.predict(input_data_reshaped)

if(prediction[0]==0):
  print("The person does not have Lung Cancer")
elif(prediction[0]==1):
  print("The person could have lower levels of Lung Cancer")
else:
  print("The person could have higher levels of lung cancer")
