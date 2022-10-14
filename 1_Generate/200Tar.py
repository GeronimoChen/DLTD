import os
import sys
import time
import tardis
import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as c
solarLumi=3.846*10**33*u.erg/u.s

homeRunDir='/scratch/user/chenxingzhuo/DLTD/ReBulk/'

IGEdens=np.genfromtxt('ProberIG/IGenhance/Density.dat',skip_header=1)
IGEelem=np.genfromtxt('ProberIG/IGenhance/IGenhanceElem.dat')
IGEvelo=IGEdens[:,1]
IGEelem[:,1:]=IGEelem[:,1:]-10**-6+10**-8
baseElem=np.zeros([6,30])

for i in range(30):
    baseElem[0,i]=IGEelem[1,i+1]
    baseElem[1,i]=IGEelem[13,i+1]
    baseElem[2,i]=IGEelem[23,i+1]
    baseElem[3,i]=IGEelem[33,i+1]
    baseElem[4,i]=IGEelem[46,i+1]
    baseElem[5,i]=IGEelem[69,i+1]

JobIndex=sys.argv[1]
ThreadIndex=sys.argv[2]
preFix=str(JobIndex)+'_'+str(ThreadIndex)

tableData=np.load('ParamIn2/'+preFix+'_Aux.npy')
elemData=np.load('ParamIn2/'+preFix+'_Elem.npy')

def RewriteOneAbund(Element,chosenelem,zone,replacer):
    if zone==0:Element[0:12,chosenelem]=replacer
    if zone==1:Element[12:22,chosenelem]=replacer
    if zone==2:Element[22:32,chosenelem]=replacer
    if zone==3:Element[32:45,chosenelem]=replacer
    if zone==4:Element[45:68,chosenelem]=replacer
    if zone==5:Element[68:109,chosenelem]=replacer
    return Element

def densMaker(dSamp):
    pivot,slope=dSamp[0],dSamp[1]
    CHdens=IGEdens.copy()
    CHdens[:,2]=CHdens[:,2]**slope
    CHdens[:,2]=CHdens[:,2]/CHdens[30,2]*IGEdens[30,2]*pivot
    return CHdens

def elemWriteFile(Element,fileName):
    Element=pd.DataFrame(Element)
    Element.to_csv(fileName,index=True,header=False,sep=' ')
    with open(fileName,'r+') as f:
        old=f.read()
        f.seek(0)
        f.write('# index Z=1 - Z=30\n')
        f.write(old)
    return

def densWriteFile(densData,fileName):
    densHeader='''11.5741 day
 #index velocity (km/s) density (g/cm^3)
'''
    pdOut=pd.DataFrame(densData)
    pdOut.to_csv(fileName,index=True,header=False,sep=' ')
    with open(fileName,'r') as reader:
        densOut=reader.read()
    with open(fileName,'w') as writter:
        writter.write(densHeader+densOut)
    return

fluxList=np.zeros([len(tableData),2000])
tempList=np.zeros([len(tableData),len(IGEdens)])
photTemp=np.zeros(len(tableData))*np.nan
for indexer in range(len(tableData)):
    YamlHere=tardis.yaml_load(homeRunDir+'ProberIG/IGenhance/IGenhance.yml')
    YamlHere['montecarlo']['nthreads']=np.random.randint(10,56)    
    YamlHere['montecarlo']['seed']=np.random.randint(678098)
    YamlHere['montecarlo']['last_no_of_packets']=np.random.randint(30,200)*10000
    YamlHere['montecarlo']['no_of_packets']=np.random.randint(15,30)*10000
    YamlHere['montecarlo']['no_of_virtual_packets']=np.random.choice([2,4,6,8])
    YamlHere['model']['abundances']['filename']=homeRunDir+'Cache/'+preFix+'.elem.dat'
    YamlHere['model']['structure']['filename']=homeRunDir+'Cache/'+preFix+'.dens.dat'
    YamlHere['supernova']['luminosity_requested']=10**tableData[indexer,0]*solarLumi
    YamlHere['supernova']['time_explosion']=tableData[indexer,1]*u.d
    YamlHere['model']['structure']['v_inner_boundary']=tableData[indexer,2]*u.km/u.s
    
    elemNew=IGEelem.copy()
    elemCh=elemData[indexer]
    for zone in range(elemCh.shape[0]):
        for elemIndex in range(1,elemCh.shape[1]+1):
            elemNew=RewriteOneAbund(elemNew,elemIndex,zone,elemCh[zone,elemIndex-1])
    densNew=densMaker(tableData[indexer,3:5])
    elemWriteFile(elemNew[:,1:],'Cache/'+preFix+'.elem.dat')
    densWriteFile(densNew[:,1:],'Cache/'+preFix+'.dens.dat')
    time.sleep(0.1)
    
    try:runOut=tardis.run_tardis(YamlHere)
    except:continue
    fluxList[indexer]=runOut.runner.spectrum_virtual.luminosity_density_lambda.value
    veloGrid=runOut.model.velocity.to('km/s').value[1:]
    startShell=np.argmin(np.abs(IGEvelo-veloGrid.min()))
    tempList[indexer,startShell:]=runOut.plasma.t_rad
    photTemp[indexer]=runOut.plasma.t_inner.value
    
    np.save('SpecOut2/'+preFix+'.spec.npy',fluxList)
    np.save('SpecOut2/'+preFix+'.temp.npy',tempList)
    np.save('SpecOut2/'+preFix+'.phot.npy',photTemp)

















