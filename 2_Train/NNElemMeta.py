import os
import sys

outter=sys.argv[1]
outter=int(outter)
modelIndex=sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[3]

import tqdm
import glob
import yaml
import keras
import astropy
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import astropy.units as u
import astropy.constants as c
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.optimizers import adam
from scipy.signal import savgol_filter
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
import keras.backend.tensorflow_backend as KTF
from dust_extinction.averages import GCC09_MWAvg
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d,UnivariateSpline
from dust_extinction.parameter_averages import CCM89,F99
from keras.callbacks import TerminateOnNaN,EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Dropout,LocallyConnected1D,AveragePooling1D,Flatten,Layer,Reshape
from keras.layers import Input,Conv1D,MaxPooling1D,BatchNormalization,Activation,Add,UpSampling1D,Concatenate

from GaussianNetworkKit import *

wave=np.genfromtxt('Prim.ascii')[:,0]

def Normalizer(spec,shortwave=6500,longwave=7500):
    small=np.argmin(abs(spec[:,0]-shortwave))
    long=np.argmin(abs(spec[:,0]-longwave))
    if small<long:spec[:,1]=spec[:,1]/np.average(spec[small:long,1])
    if small>long:spec[:,1]=spec[:,1]/np.average(spec[long:small,1])
    return spec

X_train=np.load('DataSet/110KRun/X_train.npy')
X_test=np.load('DataSet/110KRun/X_test.npy')
Y_train=np.load('DataSet/110KRun/Y_train.npy')
Y_test=np.load('DataSet/110KRun/Y_test.npy')
Yaux_train=np.load('DataSet/110KRun/Yaux_train.npy')
Yaux_test=np.load('DataSet/110KRun/Yaux_test.npy')

def MRNNSoftMaxES(CellNumber=7,outter=0,X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,Yaux_train=Yaux_train,Yaux_test=Yaux_test):
    INput=Input(shape=(X_train.shape[1],1,))
    conv1=Conv1D(8,15,strides=2,padding='same')(INput)
    conv1=Conv1D(16,3,strides=1,padding='same')(conv1)
    batc1=BatchNormalization()(conv1)
    acti1=Activation('selu')(batc1)
    pool1=MaxPooling1D(2)(acti1)
    
    conv2=Conv1D(8,1)(pool1)
    batc2=BatchNormalization()(conv2)
    acti2=Activation('selu')(batc2)
    conv3=Conv1D(16,3,padding='same')(acti2)
    
    adds=[pool1]
    
    addi=Add()(adds+[conv3])
    adds.append(addi)
    
    for i in range(CellNumber):
        conv2=Conv1D(8,1)(addi)
        batc2=BatchNormalization()(conv2)
        acti2=Activation('selu')(batc2)
        conv3=Conv1D(16,3,padding='same')(acti2)
        addi=Add()(adds+[conv3])
        adds.append(addi)
    
    batc2=BatchNormalization()(addi)
    
    flat1=keras.layers.Flatten()(batc2)
    drop1=Dropout(0.2)(flat1)
    dens1=Dense(256,activation='selu')(drop1)
    
    INput2=Input(shape=(3,))
    dens2=Dense(6,activation='selu')(INput2)
    dens2=Dense(9,activation='selu')(dens2)
    conc1=Concatenate()([INput2,dens2])
    dens2=Dense(21,activation='selu')(conc1)
    conc1=Concatenate()([INput2,dens2])
    dens2=Dense(45,activation='selu')(conc1)
    conc1=Concatenate()([INput2,dens2])
    
    conc2=Concatenate()([conc1,dens1])
    dens3=Dense(384,activation='selu')(conc2)
    drop2=Dropout(0.2)(dens3)
    
    dens3=Dense(256,activation='selu')(drop2)
    dens3=Dense(256,activation='selu')(dens3)
    dens3=Dense(256,activation='selu')(dens3)
    dens3=Dense(256,activation='selu')(dens3)
    
    if outter==6:mu,sig=GaussianLayer(2,name='ld_out')(dens3)
    else:mu,sig=GaussianLayer(30,hardMax=False,name='zone_'+str(outter)+'_out')(dens3)
    
    callbackList=[EarlyStopping(patience=10,restore_best_weights=True),
                  TerminateOnNaN(),
                  ModelCheckpoint(filepath='MdSaver/110KLogML/Model_'+str(outter)+'_'+str(modelIndex),
                                  save_weights_only=True,
                                  monitor='loss',
                                  mode='auto',
                                  save_best_only=True)]
    
    model=Model(inputs=[INput,INput2],outputs=mu)
    print(model.summary())
    opt=keras.optimizers.adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss=custom_loss(sig))
    
    if outter==6:
        mask=np.array([True,False,True,False,False])
        YtrainSel=Yaux_train[:,mask]
        YtrainSelInX=Yaux_train[:,mask==False]
        YtestSel=Yaux_test[:,mask]
        YtestSelInX=Yaux_test[:,mask==False]
    else:
        mask=np.array([True,False,True,False,False])
        YtrainSel=Y_train[:,outter,:]
        YtestSel=Y_test[:,outter,:]
        YtrainSelInX=Yaux_train[:,mask==False]
        YtestSelInX=Yaux_test[:,mask==False]
    history1=model.fit([X_train,YtrainSelInX],YtrainSel,\
                       validation_data=[[X_test,YtestSelInX],YtestSel],\
                       callbacks=callbackList,\
                       verbose=2,epochs=200,batch_size=200)
    opt=keras.optimizers.adam(lr=0.0000003,decay=1e-6)
    model.compile(optimizer=opt,loss=custom_loss(sig))
    history2=model.fit([X_train,YtrainSelInX],YtrainSel,\
                       validation_data=[[X_test,YtestSelInX],YtestSel],\
                       callbacks=callbackList,\
                       verbose=2,epochs=100,batch_size=200)
    return model,history1,history2


model,history1,history2=MRNNSoftMaxES(outter=outter)

model.save('MdSaver/110KLogML/Model_'+str(outter)+'_'+modelIndex+'.hdf')
history1=pd.DataFrame(history1.history)
history2=pd.DataFrame(history2.history)
history1.to_csv('Metric/110KLogML/ES_'+str(outter)+'_'+modelIndex+'_His1.csv')
history2.to_csv('Metric/110KLogML/ES_'+str(outter)+'_'+modelIndex+'_His2.csv')












