import os
import sys
import time
import glob
import yaml
import tardis
import astropy
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

snname=sys.argv[1]
print('Start Running. ')

solarLumi=3.846*10**33*u.erg/u.s
wave=np.genfromtxt('Prim.ascii')[:,0]
def Normalizer(spec,shortwave=6500,longwave=7500):
    small=np.argmin(abs(spec[:,0]-shortwave))
    long=np.argmin(abs(spec[:,0]-longwave))
    if small<long:spec[:,1]=spec[:,1]/np.average(spec[small:long,1])
    if small>long:spec[:,1]=spec[:,1]/np.average(spec[long:small,1])
    return spec

matBig=np.load('../3_Predict/predOutMassive/'+snname+'/matBig.npy')
errBig=np.load('../3_Predict/predOutMassive/'+snname+'/errBig.npy')
auxBig=np.load('../3_Predict/predOutMassive/'+snname+'/auxBig.npy')
aerBig=np.load('../3_Predict/predOutMassive/'+snname+'/aerBig.npy')
Xinput=np.load('../3_Predict/predOutMassive/'+snname+'/Xinput.npy')

#I Guess You Will Make Some Changes Here... 
homedir='/scratch/user/chenxingzhuo/YYTD/'
cachedir='/scratch/user/chenxingzhuo/YYTD/2_RefitSpec/Cache/'
specoutdir='/scratch/user/chenxingzhuo/YYTD/2_RefitSpec/SpecOutMassive2/'

IGEdens=np.genfromtxt(homedir+'IGenhance/Density.dat',skip_header=1)
IGEelem=np.genfromtxt(homedir+'IGenhance/IGenhanceElem.dat')
IGEvelo=IGEdens[:,1]
IGEelem[:,1:]=IGEelem[:,1:]-10**-6+10**-8

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
    time.sleep(0.1)
    with open(fileName,'r+') as f:
        old=f.read()
        f.seek(0)
        f.write('# index Z=1 - Z=30\n')
        f.write(old)
    #time.sleep(0.1)
    return
def densWriteFile(densData,fileName):
    densHeader='''11.5741 day
 #index velocity (km/s) density (g/cm^3)
'''
    pdOut=pd.DataFrame(densData)
    with open(fileName,'w') as writter:
        writter.write(densHeader+pdOut.to_string(header=False))
    #time.sleep(0.1)
    return
def yamlWritter(lumi,expTime,velo,cacheFile,yamlTempFile=homedir+'/IGenhance/IGenhance.yml',):
    YamlHere=tardis.io.util.yaml_load_file(yamlTempFile)
    #And You Will Make Some Changes Here... 
    YamlHere['atom_data']='/scratch/user/chenxingzhuo/TDAN/OneTrial/kurucz_cd23_chianti_H_He.h5'
    
    YamlHere['montecarlo']['nthreads']=30
    YamlHere['montecarlo']['seed']=np.random.randint(36789212)
    YamlHere['montecarlo']['last_no_of_packets']=300000
    YamlHere['montecarlo']['no_of_packets']=30000
    YamlHere['montecarlo']['no_of_virtual_packets']=3
    
    YamlHere['model']['abundances']['filename']=cacheFile+'.elem.dat'
    YamlHere['model']['structure']['filename']=cacheFile+'.dens.dat'
    YamlHere['model']['structure']['v_inner_boundary']=str(velo)+' km/s'

    YamlHere['supernova']['luminosity_requested']=str(lumi)+' log_lsun'
    YamlHere['supernova']['time_explosion']=str(expTime)+' day'
    with open(cacheFile+'.yml','w') as writter:yaml.dump(YamlHere,stream=writter)
    return

def CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFluxIn,matBigIn=matBig,auxBigIn=auxBig):
    pivotGOne=np.array([pivotCen-pivotDif,pivotCen,pivotCen+pivotDif])
    slopeGOne=np.array([slopeCen-slopeDif,slopeCen,slopeCen+slopeDif])

    comList=''
    for pivot in pivotGOne:
        pivot=round(pivot,3)
        for slope in slopeGOne:
            slope=round(slope,3)
            auxIndexer=np.argwhere((auxBigIn[:,:,0,3]==pivot)&(auxBigIn[:,:,0,4]==slope))[0]
            auxOne=auxBigIn[auxIndexer[0],auxIndexer[1]]
            for specIndex in range(len(auxOne)):
                print(auxIndexer,specIndex)
                print(FitFluxIn[auxIndexer[0],auxIndexer[1],specIndex])
                if np.isnan(np.mean(FitFluxIn[auxIndexer[0],auxIndexer[1],specIndex]))==False and np.mean(FitFluxIn[auxIndexer[0],auxIndexer[1],specIndex])!=0:
                    continue
                lumi=auxOne[specIndex,0]
                velo=auxOne[specIndex,2]
                expTime=auxOne[specIndex,1]
                #print(specIndex)
                cacheFile=cachedir+snname+'_Pivot_'+str(pivot)+'_Slope_'+str(slope)+'_Index_'+str(specIndex)
                yamlWritter(lumi,expTime,velo,cacheFile=cacheFile)

                elemNew=IGEelem.copy()
                elemCh=matBigIn[auxIndexer[0],auxIndexer[1],specIndex]
                for zone in range(elemCh.shape[0]):
                    for elemIndex in range(1,elemCh.shape[1]+1):
                        elemNew=RewriteOneAbund(elemNew,elemIndex,zone,elemCh[zone,elemIndex-1])
                densNew=densMaker([pivot,slope])
                elemWriteFile(elemNew[:,1:],cacheFile+'.elem.dat')
                densWriteFile(densNew[:,1:],cacheFile+'.dens.dat')

                comList=comList+'python tardisWrap.py '+cacheFile+'.yml False \n' #ulimit -v 27000000 && 
    return comList,pivotGOne,slopeGOne

def BestSelecter(FitFluxIn,ObsSpecIn,matBigIn=matBig,auxBigIn=auxBig):
    pivotList=[]
    slopeList=[]
    chisqList=[]
    for pivInd in range(FitFluxIn.shape[0]):
        for sloInd in range(FitFluxIn.shape[1]):
            fluxList=FitFluxIn[pivInd,sloInd]
            if np.isnan(np.sum(fluxList)):continue
            pivot=auxBigIn[pivInd,sloInd,0,3]
            slope=auxBigIn[pivInd,sloInd,0,4]
            pivotList.append(round(float(pivot),3))
            slopeList.append(round(float(slope),3))
            chisqOneSeq=0
            for j in range(len(ObsSpecIn)):
                fluxObs=ObsSpecIn[j][:,0]
                fluxFit=fluxList[j]
                mask=(fluxObs>0)
                fluxObs=fluxObs[mask]
                fluxFit=fluxFit[mask]
                fluxObs=fluxObs/np.mean(fluxObs)
                fluxFit=fluxFit/np.mean(fluxFit)
                chisqOneSeq=chisqOneSeq+np.sum((fluxObs-fluxFit)**2)
            chisqList.append(chisqOneSeq)
    chisqList=np.array(chisqList)
    print(chisqList)
    print(pivotList)
    print(slopeList)
    minHere=np.nanargmin(chisqList)
    pivotCen=pivotList[minHere]
    slopeCen=slopeList[minHere]
    return pivotCen,slopeCen

def CommandRunner(comListIn):
    with open('Cache/Runner_'+snname+'.in','w') as writter:
        writter.write(comListIn)
    #Probably you don't have a tamulauncher command that balances the load of multiple machines on a supercomputer, so please make some changes here.... 
    com=os.popen('tamulauncher --norelease-resources --commands-pernode 2 '+'Cache/Runner_'+snname+'.in')
    print(com.read())
    #com=os.popen('tamulauncher --remove-logs '+'Cache/Runner_'+snname+'.in')
    #print(com.read())
    return 
def SpecCollector(FitFluxIn,pivotGOneIn,slopeGOneIn):
    for pivot in pivotGOneIn:
        pivot=round(pivot,3)
        for slope in slopeGOneIn:
            slope=round(slope,3)
            auxIndexer=np.argwhere((auxBig[:,:,0,3]==pivot)&(auxBig[:,:,0,4]==slope))[0]
            auxOne=auxBig[auxIndexer[0],auxIndexer[1]]
            for specIndex in range(len(auxOne)):
                if np.isnan(np.sum(FitFluxIn[auxIndexer[0],auxIndexer[1],specIndex]))==False:continue
                cacheFile=cachedir+snname+'_Pivot_'+str(pivot)+'_Slope_'+str(slope)+'_Index_'+str(specIndex)
                try:flux=np.load(cacheFile+'.flux.npy')
                except:continue
                FitFluxIn[auxIndexer[0],auxIndexer[1],specIndex]=flux
    np.save(specoutdir+snname+'_FitFlux.npy',FitFluxIn)
    return FitFluxIn

if os.path.exists(specoutdir+snname+'_FitFlux.npy'):FitFlux=np.load(specoutdir+snname+'_FitFlux.npy')
else:FitFlux=np.zeros([matBig.shape[0],matBig.shape[1],matBig.shape[2],2000])*np.nan

pivotCen=1.1
slopeCen=1.1
pivotDif=0.6
slopeDif=0.6
comList,pivotGOne,slopeGOne=CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFlux)

if len(comList)>0:
    print('Start 2')
    CommandRunner(comList)
    FitFlux=SpecCollector(FitFlux,pivotGOne,slopeGOne)
else:print('Already Did. ')

pivotCen,slopeCen=BestSelecter(FitFlux,Xinput)
pivotDif=0.2
slopeDif=0.2
comList,pivotGOne,slopeGOne=CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFlux)

if len(comList)>0:
    CommandRunner(comList)
    FitFlux=SpecCollector(FitFlux,pivotGOne,slopeGOne)
else:print('Already Did. ')

pivotCen,slopeCen=BestSelecter(FitFlux,Xinput)
pivotDif=0.2
slopeDif=0.2
comList,pivotGOne,slopeGOne=CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFlux)
if len(comList)>0:
    CommandRunner(comList)
    FitFlux=SpecCollector(FitFlux,pivotGOne,slopeGOne)
else:print('Already Did.')

pivotCen,slopeCen=BestSelecter(FitFlux,Xinput)
pivotDif=0.1
slopeDif=0.1
comList,pivotGOne,slopeGOne=CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFlux)
if len(comList)>0:
    CommandRunner(comList)
    FitFlux=SpecCollector(FitFlux,pivotGOne,slopeGOne)
else:print('Already Did. ')

pivotCen,slopeCen=BestSelecter(FitFlux,Xinput)
pivotDif=0.1
slopeDif=0.1
comList,pivotGOne,slopeGOne=CommandPrepare(pivotCen,slopeCen,pivotDif,slopeDif,FitFlux)
if len(comList)>0:
    CommandRunner(comList)
    FitFlux=SpecCollector(FitFlux,pivotGOne,slopeGOne)
else:print('Already Did. ')
com=os.popen('rm Cache/'+snname+'_*')
