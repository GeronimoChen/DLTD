import os
import sys
import time
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
from keras.layers import Layer
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from dust_extinction.averages import GCC09_MWAvg
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d,UnivariateSpline
from dust_extinction.parameter_averages import CCM89,F99
from keras.callbacks import TerminateOnNaN,EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Dropout,LocallyConnected1D,AveragePooling1D,Flatten,Layer,Reshape
from keras.layers import Input,Conv1D,MaxPooling1D,BatchNormalization,Activation,Add,UpSampling1D,Concatenate

snName=sys.argv[1]
specDir=sys.argv[2]
predOutDir=sys.argv[3]
networkDir=sys.argv[4]
ebvHost=float(sys.argv[5])
ebvMw=float(sys.argv[6])
redshift=float(sys.argv[7])

def gaussian_loss(y_true, y_pred):
    return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.divide(tf.square(y_true - y_pred), sigma)) + 1e-6
def custom_loss(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.divide(tf.square(y_true - y_pred), sigma)) + 1e-6
    return gaussian_loss
class GaussianLayer(Layer):    
    def __init__(self, output_dim=30,hardMax=False, **kwargs):
        self.output_dim = output_dim
        self.hardMax=hardMax
        super(GaussianLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                      shape=(256, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                      shape=(256, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(GaussianLayer, self).build(input_shape)
    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        if self.hardMax==True:
            output_mu=K.relu(output_mu)
            output_mu=output_mu/K.sum(output_mu,axis=-1,keepdims=True)#
            #output_mu=K.softmax(output_mu)
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]

wave=np.genfromtxt('Prim.ascii')[:,0]
YauxNorm=np.load('YauxNorm.npy')
#dm15List=pd.read_csv('dm15_Example.csv',index_col=0)

def Normalizer(spec,shortwave=6500,longwave=7500):
    small=np.argmin(abs(spec[:,0]-shortwave))
    long=np.argmin(abs(spec[:,0]-longwave))
    if small<long:spec[:,1]=spec[:,1]/np.average(spec[small:long,1])
    if small>long:spec[:,1]=spec[:,1]/np.average(spec[long:small,1])
    return spec
def windowSpec(spec):
    spFunc=interp1d(spec[:,0],spec[:,1],fill_value=np.nan,bounds_error=False)
    smFlux=spFunc(wave)
    smFlux=smFlux/np.nanmean(smFlux)
    smFlux[np.isnan(smFlux)]=-1
    return np.array([wave,smFlux]).T
def specListMaker(starTableDir,redshift,extMW=0,extHost=0,telNameGetter=False):
    starTable=pd.read_csv(starTableDir+'starTable.csv')#'../ObserveSpectra/GoodSpec/SN2011fe/starTable.csv'
    X_snemo=[]
    timeCollect=[]
    telNameList=[]
    for i in range(len(starTable)):
        if starTable['Phase'][i]>20 or starTable['Phase'][i]<-10:continue
        fileName=starTable['Ascii file'].iloc[i]
        fileName=glob.glob(starTableDir+fileName.split('.dat')[0]+'*')[0]
        spec=np.genfromtxt(fileName)
        spec[:,1]=spec[:,1]/GCC09_MWAvg().extinguish(spec[:,0]*u.Angstrom,Ebv=extMW)
        spec[:,0]=spec[:,0]/(1+redshift)
        spec[:,1]=spec[:,1]/F99().extinguish(spec[:,0]*u.Angstrom,Ebv=extHost)
        spec=Normalizer(spec)
        spResa=windowSpec(spec)
        if np.max(spResa[:,1])>20:continue
        X_snemo.append(spResa[:,1].reshape([2000,1]))
        timeCollect.append(starTable['Phase'][i])
        if telNameGetter:
            telNameList.append(starTable['Tel'][i])
    X_snemo=np.array(X_snemo)
    timeCollect=np.array(timeCollect)+19
    if telNameGetter:
        return X_snemo,timeCollect,telNameList
    else:
        return X_snemo,timeCollect

def MRNNSoftMaxESMake(CellNumber=7,outter=0):
    INput=Input(shape=(2000,1,))
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
    model=Model(inputs=[INput,INput2],outputs=mu)
    print(model.summary())
    opt=keras.optimizers.Adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss=custom_loss(sig))
    return model

intermediateModels=[]
for outter in range(7):
    valiLossList=[]
    for mdInd in range(10):
        try:
            his2=pd.read_csv(networkDir+'/ES_'+str(outter)+'_'+str(mdInd)+'_His2.csv')
            valiLoss=his2['val_loss'].iloc[-1]
            if np.isnan(valiLoss):valiLoss=np.inf
        except:valiLoss=np.inf
        valiLossList.append(valiLoss)
    mdIndChos=np.argmin(valiLossList)
    model=MRNNSoftMaxESMake(outter=outter)
    model.load_weights(networkDir+'/Model_'+str(outter)+'_'+str(mdIndChos)+'.hdf')
    if outter==6:
        outLayerName='ld_out'
    else:outLayerName='zone_'+str(outter)+'_out'
    intermediateModels.append(K.function(inputs=[model.input[0],model.input[1]],outputs=model.get_layer(outLayerName).output))

def ModelPredictor(modelList,specList,auxInList):
    matExport=np.zeros([len(specList),6,30])
    errExport=np.zeros([len(specList),6,30])
    for j in range(6):
        mu,sigma=modelList[j]([specList,auxInList])
        matExport[:,j,:]=mu
        errExport[:,j,:]=sigma**0.5
    j=6
    mu,sigma=modelList[j]([specList,auxInList])
    sigma=sigma**0.5
    mu[:,0]=mu[:,0]*YauxNorm[1,0]+YauxNorm[0,0]
    mu[:,1]=mu[:,1]*YauxNorm[1,2]+YauxNorm[0,2]
    sigma[:,0]=sigma[:,0]*YauxNorm[1,0]
    sigma[:,1]=sigma[:,1]*YauxNorm[1,2]
    return matExport,errExport,mu,sigma

X_snemo,timeCollect=specListMaker(specDir+'/'+snName+'/',redshift,extHost=ebvHost,extMW=ebvMw)

densPivotRange=np.arange(0.2,2.01,0.1)
densSlopeRange=np.arange(0,2.01,0.1)

os.popen('mkdir -p '+predOutDir+'/'+snName)
matBig=np.zeros([len(densPivotRange),len(densSlopeRange),len(X_snemo),6,30])
errBig=np.zeros([len(densPivotRange),len(densSlopeRange),len(X_snemo),6,30])
auxBig=np.zeros([len(densPivotRange),len(densSlopeRange),len(X_snemo),5])
aerBig=np.zeros([len(densPivotRange),len(densSlopeRange),len(X_snemo),5])
for densNInd,densN in enumerate(densPivotRange):
    for densDInd,densD in enumerate(densSlopeRange):
        densN=float(round(densN,3))
        densD=float(round(densD,3))
        timeNorm=(timeCollect-YauxNorm[0,1])/YauxNorm[1,1]
        timeNorm=timeNorm.reshape([-1,1])
        dens1Norm=(np.ones([len(X_snemo),1])*densN-YauxNorm[0,3])/YauxNorm[1,3]
        dens2Norm=(np.ones([len(X_snemo),1])*densD-YauxNorm[0,4])/YauxNorm[1,4]
        auxIn=np.hstack([timeNorm,dens1Norm,dens2Norm])
        matExport,errExport,mu,sigma=ModelPredictor(intermediateModels,X_snemo,auxIn)

        auxList=np.zeros([len(X_snemo),5])
        auxList[:,0]=mu[:,0]
        auxList[:,2]=mu[:,1]
        auxList[:,1]=timeCollect
        auxList[:,3]=densN
        auxList[:,4]=densD
        sigList=np.zeros([len(X_snemo),5])
        sigList[:,0]=sigma[:,0]
        sigList[:,2]=sigma[:,1]
        matBig[densNInd,densDInd]=matExport
        errBig[densNInd,densDInd]=errExport
        auxBig[densNInd,densDInd]=auxList
        aerBig[densNInd,densDInd]=sigList
matBig=10**matBig
np.save(predOutDir+'/'+snName+'/matBig.npy',matBig)
np.save(predOutDir+'/'+snName+'/errBig.npy',errBig)
np.save(predOutDir+'/'+snName+'/auxBig.npy',auxBig)
np.save(predOutDir+'/'+snName+'/aerBig.npy',aerBig)
np.save(predOutDir+'/'+snName+'/Xinput.npy',X_snemo)




