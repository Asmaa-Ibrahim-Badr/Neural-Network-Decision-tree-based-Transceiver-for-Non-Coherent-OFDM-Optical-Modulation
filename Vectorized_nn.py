import pandas as pd
import numpy as np
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
from numpy import random
import scipy as sc
from commpy.modulation import QAMModem
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from numpy import array
from numpy.random import uniform
from numpy import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers import PReLU
# generate random integer values
iterations = 10000# size of trained  data
subcarriers_data = 3 # size of input data
subcarriers=2*(subcarriers_data+1)
q16 = QAMModem(16)
x= np.random.randint(2, size=(4*subcarriers_data*iterations)) # 16qam symbols
y =q16.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.around(ifft_sig_clipped,2)
output_y=np.around(ifft_sig,2)
in_dim = input_x.shape[1]
out_dim = output_y.shape[1]
xtrain, xtest, ytrain, ytest=train_test_split(input_x, output_y, test_size=0.1)
print(xtest.shape)
###################define model
t = time.time()
model = Sequential()
model.add(Dense(100, input_dim=in_dim))
model.add(Dense(32))
model.add(PReLU())
model.add(Dense(32))
model.add(PReLU())
model.add(Dense(32))
model.add(PReLU())
model.add(Dense(out_dim,activation="linear"))
model.compile(loss="mse", optimizer="adam")#model.add(Dense(64,activation="tanh"))
###fit the model with train data.
model.fit(xtrain, ytrain, epochs=1000, batch_size=12, verbose=0)
#####pediction
ypred = model.predict(xtest)
ypred1 = model.predict(xtrain)
#######
x_ax = range(xtest.shape[1])
# serialize model to JSON
model_json = model.to_json()
with open("model_4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
elapsed = time.time() - t
model.save_weights("model_4.h5")
print("Saved model to disk")
print(elapsed)
plt.plot(x_ax, ypred[0,:], label="y1-test")
plt.plot(x_ax, ytest[0,:], label="y1-pred")
plt.plot(x_ax, ypred1[0,:], label="y1-test")
plt.plot(x_ax, ytrain[0,:], label="y1-pred")
#plt.scatter(x_ax, ytest[1,:],  s=6, label="y2-test")
#plt.plot(x_ax, ypred[1,:], label="y2-pred")
plt.legend()
plt.show()
#print(ypred[0,:],ytest[0,:])