import re
import statistics
import matplotlib.pyplot as plt
from numpy import random
import scipy as sc
from commpy.modulation import QAMModem
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import statistics
import matplotlib.pyplot as plt
from graphviz import Digraph
import pydotplus
from numpy import random
from tensorflow import keras
from keras.models import Input, Model
from keras.models import Sequential
from keras.layers import Dense,Concatenate
from keras.models import model_from_json
from numpy import array
from numpy.random import uniform
from numpy import hstack
from sklearn.model_selection import train_test_split
from keras.layers import PReLU
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle
from decimal import Decimal
from fxpmath import Fxp
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# generate random integer values
iterations = 100000# size of trained  data
subcarriers_data = 3 # size of input data
subcarriers=2*(subcarriers_data+1)
q16 = QAMModem(8)
x= np.random.randint(2, size=(3*subcarriers_data*iterations)) # 16qam symbols
y =q16.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.real(ifft_sig_clipped)
output_y=np.real(ifft_sig)
in_dim = ifft_sig_clipped.shape[1]
out_dim = ifft_sig.shape[1]
xtrain, xtest, ytrain, ytest=train_test_split(input_x, output_y, test_size=0.1)
print(xtest.shape)
##########################################model
model = RandomForestRegressor()
#model1 = MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=300))
model.fit(xtrain, ytrain)
#####pediction
ypred = model.predict(xtest)
ypred1 = model.predict(xtrain)
#######
x_ax = range(xtest.shape[1])
# serialize model to JSON
pickle.dump(model, open('model_forest','wb'))
print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, ypred))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, ypred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print("Saved model to disk")
plt.plot(x_ax, ypred[0,:], label="y_pred-test")
plt.plot(x_ax, ytest[0,:], label="y_test")
plt.plot(x_ax, ypred1[0,:], label="y_pred-train")
plt.plot(x_ax, ytrain[0,:], label="y_train")
#plt.scatter(x_ax, ytest[1,:],  s=6, label="y2-test")
#plt.plot(x_ax, ypred[1,:], label="y2-pred")
plt.legend()
plt.show()
#print(ypred[0,:],ytest[0,:])