import numpy as np
import pickle


input_data=(73,1,5,6,6,5,6,5,6,5,8,5,5,5,4,3,6,2,1,2,1,6,2)


#loading the saved model
loaded_model = pickle.load(open("lung_model.sav", 'rb'))


#change the data to a numpy array
input_data_as_numpy_array=np.array(input_data, dtype=float)

#reshape the numpy array as we are prediction for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

if(prediction[0]==0):
  print("The person does not have Lung Cancer")
elif(prediction[0]==1):
  print("The person could have lower levels of Lung Cancer")
else:
  print("The person could have higher levels of lung cancer")
