#Afaf Taik 
#GEI723
#No forget gate  

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.layers import RNN, Activation,LSTM
from keras.models import Model
import keras
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#basic layer on which we'll apply modifications on the gates to test the performance
class NFGLSTM(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [units, units]
        super(NFGLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_xi = self.add_weight(name='w_xi',
                                    shape=(input_shape[-1], self.units), initializer='uniform')
        
        self.w_xo = self.add_weight(name='w_xo',
                                    shape=(input_shape[-1], self.units), initializer='uniform')
        self.w_xc = self.add_weight(name='w_xc',
                                    shape=(input_shape[-1], self.units), initializer='uniform')
        self.w_hi = self.add_weight(name='w_hi',
                                    shape=(self.units, self.units), initializer='uniform')
        
        self.w_ho = self.add_weight(name='w_ho',
                                    shape=(self.units, self.units), initializer='uniform')
        self.w_hc = self.add_weight(name='w_hc',
                                    shape=(self.units, self.units), initializer='uniform')
        self.b_i = self.add_weight(name='b_i',
                                    shape=(1, self.units), initializer='zeros')
        self.b_o = self.add_weight(name='b_o',
                                    shape=(1, self.units), initializer='zeros')
        self.b_c = self.add_weight(name='b_c',
                                    shape=(1, self.units), initializer='zeros')

        self.built = True

    def call(self, x, states):
        h, c = states
        i = K.sigmoid(K.dot(x, self.w_xi) + K.dot(h, self.w_hi) + self.b_i)
        f = 1.0
        o = K.sigmoid(K.dot(x, self.w_xo) + K.dot(h, self.w_ho) + self.b_o)

        c_in = K.tanh(K.dot(x, self.w_xc) + K.dot(h, self.w_hc) + self.b_c)
        c_n = f * c + i * c_in
        h_n = o * K.tanh(c_n)

        return h_n, [h_n, c_n]


