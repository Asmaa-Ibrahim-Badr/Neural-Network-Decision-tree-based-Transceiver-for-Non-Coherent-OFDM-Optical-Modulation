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
# generate random integer values
iterations = 10000# size of trained  data
subcarriers_data = 7# size of input data
subcarriers=2*(subcarriers_data+1)
q16 = QAMModem(4)
total_bits=2*subcarriers_data*iterations
x= np.random.randint(2, size=(total_bits)) # 16qam symbols
y =q16.modulate(x)
y=np.reshape(y,(iterations,subcarriers_data))
opt_signal=np.concatenate((np.zeros((iterations,1)),y,np.zeros((iterations,1)),np.flip(y.conj(),axis=1)),axis=1)
ifft_sig=np.fft.ifft(opt_signal)
ifft_sig_clipped=(ifft_sig+abs(ifft_sig))/2
input_x=np.around(ifft_sig_clipped,2)
output_y=np.around(ifft_sig,2)
# load json and create model
json_file = open('model_4q_16sub_128nn_3layer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_4q_16sub_128nn_3layer.h5")
#####pediction
#loaded_model.compile(loss="mse", optimizer="adam")
SNR=list(range(0,100,10))
BER=list(range(0,100,10))
for i in range(10):
    SNR_amp=10**(SNR[i]//10)
    signal_power=np.mean(input_x*input_x,1)
    noise_power=signal_power/SNR_amp
    noise_amp=np.sqrt(noise_power)
    noise_rand=np.random.normal(0, 1, size=(iterations,subcarriers))
    noise=noise_rand * noise_amp[:, np.newaxis]
    input_x_noisy=input_x+noise
    subcarriers_index=subcarriers_data+1
    ypred = loaded_model.predict(input_x_noisy)
#loaded_model.evaluate(output_y, ypred)
    fft_data = np.around(ypred,2)
    fft_sig1 = np.fft.fft(fft_data)
    fft_sig=fft_sig1[:,1:subcarriers_index]
    fft_sig=fft_sig.reshape(-1)
    rx_demod = q16.demodulate(fft_sig,"hard")
    Out_bits = np.array(rx_demod)
    In_bits = np.array(x)
    correct=sum(np.equal(In_bits,Out_bits))
    BER[i]=(total_bits-correct)/(total_bits)
x_ax = range(10)
mse=mean_squared_error(np.real(output_y),np.real(ypred))
plt.semilogy(x_ax, BER, label="BER")
plt.legend()
plt.show()
print(BER)
print(mse)
