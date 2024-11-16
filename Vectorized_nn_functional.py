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
from graphviz import Digraph
import pydotplus
import time
from numpy import random
import scipy as sc
from commpy.modulation import QAMModem
import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model
from keras.models import Sequential
from keras.layers import Dense,Concatenate
from keras.models import model_from_json
from numpy import array
from numpy.random import uniform
from numpy import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers import PReLU
# generate random integer values
iterations = 10000# size of trained  data
subcarriers_data = 7 # size of input data
subcarriers=2*(subcarriers_data+1)
q16 = QAMModem(4)
x= np.random.randint(2, size=(2*subcarriers_data*iterations)) # 16qam symbols
y =q16.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.around(ifft_sig_clipped,2)
output_y=np.around(ifft_sig,2)
in_dim = input_x.shape[1]
out_dim = output_y.shape[1]
SNR=40
SNR_amp=10**(SNR//10)
signal_power=np.mean(input_x*input_x,1)
noise_power=signal_power/SNR_amp
noise_amp=np.sqrt(noise_power)
noise_rand=np.random.normal(0, 1, size=(iterations,subcarriers))
noise=noise_rand * noise_amp[:, np.newaxis]
input_x_noisy=input_x+noise
xtrain, xtest, ytrain, ytest=train_test_split(input_x, output_y, test_size=0.1)
print(xtest.shape)
###################define model
t=time.time()
inputs = Input(shape=(in_dim,))
layer1 = Dense(in_dim)(inputs)
layer_prelu = PReLU()(layer1)
layer2 = Dense(128)(layer_prelu)
layer2_prelu = PReLU()(layer2)
layer2_in = Dense(128)(inputs)
layer2_prelu_in = PReLU()(layer2_in)
layer2_concatted = Concatenate()([layer2_prelu_in, layer2_prelu])
layer3 = Dense(128)(layer2_concatted)
layer3_prelu = PReLU()(layer3)
layer3_in = Dense(128)(inputs)
layer3_prelu_in = PReLU()(layer3_in)
layer3_concatted = Concatenate()([layer3_prelu_in, layer3_prelu])
layer4_prelu = PReLU()(layer3_concatted)
layer4_in = Dense(128)(inputs)
layer4_prelu_in = PReLU()(layer4_in)
layer4_concatted = Concatenate()([layer4_prelu_in, layer4_prelu])
layer5_prelu = PReLU()(layer4_concatted)
layer5_in = Dense(128)(inputs)
layer5_prelu_in = PReLU()(layer5_in)
layer5_concatted = Concatenate()([layer5_prelu_in, layer5_prelu])
out_layer = Dense(out_dim,activation="linear")(layer4_concatted)
logistic_model = Model(inputs, out_layer)
logistic_model.compile(loss="mse", optimizer="adam")#model.add(Dense(64,activation="tanh"))
print(logistic_model.summary())
#keras.utils.plot_model(logistic_model, "my_first_model_with_shape_info.png", show_shapes=True)
###fit the model with train data.
logistic_model.fit(xtrain, ytrain, epochs=1000, batch_size=12, verbose=0)
#####pediction
ypred = logistic_model.predict(xtest)
ypred1 = logistic_model.predict(xtrain)
#######
x_ax = range(xtest.shape[1])
# serialize model to JSON
#model_json = logistic_model.to_json()
#with open("model_4q_128sub_128nn.json", "w") as json_file:
 #   json_file.write(model_json)
# serialize weights to HDF5
#logistic_model.save_weights("model_4q_128sub_128nn.h5")
elapsed = time.time() - t
print("Saved model to disk")
print(elapsed)
mse=mean_squared_error(ytest,ypred)
#plt.plot(x_ax, ypred[0,:], label="y1-test")
#plt.plot(x_ax, ytest[0,:], label="y1-pred")
#plt.plot(x_ax, ypred1[0,:], label="y1-test")
#plt.plot(x_ax, ytrain[0,:], label="y1-pred")
#plt.scatter(x_ax, ytest[1,:],  s=6, label="y2-test")
#plt.plot(x_ax, ypred[1,:], label="y2-pred")
#plt.legend()
#plt.show()
print(mse)