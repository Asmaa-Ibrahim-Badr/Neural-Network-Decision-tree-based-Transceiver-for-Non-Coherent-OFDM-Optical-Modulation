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
import pickle
# generate random integer values
iterations = 1000# size of trained  data
subcarriers_data = 3 # size of input data
subcarriers=2*(subcarriers_data+1)
q4 = QAMModem(8)
x= np.random.randint(2, size=(3*subcarriers_data*iterations)) # 16qam symbols
y =q4.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.real(ifft_sig_clipped)
output_y=np.real(ifft_sig)
SNR=40
SNR_amp=10**(SNR//10)
signal_power=np.mean(input_x*input_x,1)
noise_power=signal_power/SNR_amp
noise_amp=np.sqrt(noise_power)
noise_rand=np.random.normal(0, 1, size=(iterations,subcarriers))
noise=noise_rand * noise_amp[:, np.newaxis]
input_x_noisy=input_x+noise
# load json and create model
loaded_model =pickle.load(open('model_forest', 'rb'))
#####pediction
#loaded_model.compile(loss="mse", optimizer="adam")
subcarriers_index=subcarriers_data+1
ypred = loaded_model.predict(input_x)
x_ax = range(ypred.shape[1])
#loaded_model.evaluate(output_y, ypred)
fft_data = np.around(ypred,2)
fft_sig1 = np.fft.fft(fft_data)
fft_sig=fft_sig1[:,1:subcarriers_index]
fft_sig=fft_sig.reshape(-1)
rx_demod = q4.demodulate(fft_sig,"hard")
Out_bits = np.array(rx_demod)
In_bits = np.array(x)
correct=sum(np.equal(In_bits,Out_bits))
BER=(3*subcarriers_data*iterations-correct)/(3*subcarriers_data*iterations)
print (BER)
print(input_x_noisy,input_x)