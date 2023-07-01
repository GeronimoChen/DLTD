# The AIAI Package

This notebook introduces how to establish the working environment and run the AIAI fitting program for a type Ia supernova spectral time sequence. 

## The Environment

A Linux system is strongly recommended. The installation of the program will be based on an existing TARDIS environment. Please refer to the website "https://tardis-sn.github.io/tardis/installation.html" when installing TARDIS. 

## Installation

The program has integrated the neural network prediction part and the spectral fitting part of the whole AIAI project, the code of which are illustrated in the folder "3_Predict" and "4_Fit". 

To run the neural network prediction part, the following packages are required: 

- tensorflow
- dust_extinction
- pandas
- tqdm

To run the spectral fitting part, the following packages are required: 

- tardis 

pip install AIAISN

## Download The Data

We use kaggle platform to store the files which are too large to be uploaded to github. Before downloading the data, please: 

1. Install the kaggle package with "pip install kaggle" command. 
2. Create an account on the kaggle platform, it should be free. 
3. Follow the instructions on the kaggle website "https://www.kaggle.com/docs/api". 
4. Run the downloading script with "./data_download.sh". 

Moreover, don't forget to download the TARDIS atomic data from this address: "https://github.com/tardis-sn/tardis-refdata/raw/master/atom_data/kurucz_cd23_chianti_H_He.h5". 


## Running a Supernova Fitting

### Prepare the supernova spectral time sequence



### Predict the supernova element abundance

The neural network prediction function is: 

from AIAISN import predictFunc
predictFunc.readNetPredictSave(snName='SN2011fe',specDir='ObserveExample/',predOutDir='predOutExample/',networkDir='../2_Train/MdSaver/110KLogML/',ebvHost=0,ebvMw=0.0088248,redshift=0.000804)



### Fit the spectra for a density

from AIAISN import fittingFunc
fittingFunc.fitRunner(snname='SN2011fe',predOutDir='predOutExample/',specoutdir='specOutExample/',atomData='/[path_to]/kurucz_cd23_chianti_H_He.h5',cachedir='Cache/',mpiCommandor='',threadCount=8)


tamulauncher --norelease-resources --commands-pernode 2 








python Predictor.py SN2011fe ObserveExample/ predOutExample/ /home/gesa/SuperCode/GithubUpload/2_Train/MdSaver/110KLogML/ 0 0.00882 0.000804
python Fitting.py SN2011fe predOutExample/ specOutExample/ kurucz_cd23_chianti_H_He.h5


