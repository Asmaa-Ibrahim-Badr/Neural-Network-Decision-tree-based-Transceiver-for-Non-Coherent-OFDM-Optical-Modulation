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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle
from decimal import Decimal
from fxpmath import Fxp
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from microlearn.offloader import Offload
import tempfile
import os
import subprocess
# generate random integer values
iterations = 1000# size of trained  data
subcarriers_data = 15 # size of input data
subcarriers=2*(subcarriers_data+1)
q4 = QAMModem(4)
x= np.random.randint(2, size=(2*subcarriers_data*iterations)) # 16qam symbols
y =q4.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.real(ifft_sig_clipped)
output_y=np.real(ifft_sig)
in_dim = ifft_sig_clipped.shape[1]
out_dim = ifft_sig.shape[1]
SNR=40
SNR_amp=10**(SNR//10)
signal_power=np.mean(input_x*input_x,1)
noise_power=signal_power/SNR_amp
noise_amp=np.sqrt(noise_power)
noise_rand=np.random.normal(0, 1, size=(iterations,subcarriers))
noise=noise_rand * noise_amp[:, np.newaxis]
input_x_noisy=input_x+noise
xtrain, xtest, ytrain, ytest=train_test_split(input_x_noisy, output_y, test_size=0.1)
print(xtest.shape)
##########################################model
model1= DecisionTreeRegressor(max_depth=20)
model = MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=30))
model.fit(xtrain, ytrain)
#####pediction
ypred = model.predict(xtest)
ypred1 = model.predict(xtrain)
#######
x_ax = range(xtest.shape[1])
# serialize model to JSON
pickle.dump(model, open('model_tree15_noise20db','wb'))
# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
model.save('export_path/model')

print('\nSaved model:')
print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, ypred))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, ypred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print("Saved model to disk")
