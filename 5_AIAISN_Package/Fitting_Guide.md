# The AIAI Package

This notebook introduces how to establish the working environment and run the AIAI fitting program for a type Ia supernova spectral time sequence. 

## The Environment

A Linux system is strongly recommended. The installation of the program will be based on an existing TARDIS environment. Please refer to the website "https://tardis-sn.github.io/tardis/installation.html" when installing TARDIS. 

## Installation

The program has integrated the neural network prediction part and the spectral fitting part of the whole AIAI project, the code of which are illustrated in the folder "3_Predict" and "4_Fit". 

To run the neural network prediction part, the following packages are required: 

- tensorflow == 2.12.0
- dust_extinction == 1.1
- pandas == 1.5.3
- tqdm == 4.64.1

To run the spectral fitting part, the following packages are required: 

- tardis == 2023.1.19.0.dev3+gde912971

The package dependencies are not encoded in the AIAISN package installation, and the package versions presented here are not the only versions suitable to the AIAISN package. 

The command to install the AIAISN package is: 

'''
pip install AIAISN
'''

or 

'''
pip install --upgrade --index-url https://test.pypi.org/simple/ AIAISN
'''

## Download The Data

We use kaggle platform to store the files which are too large to be uploaded to github. Before downloading the data, please: 

1. Install the kaggle package with "pip install kaggle" command. 
2. Create an account on the kaggle platform, it should be free. 
3. Follow the instructions on the kaggle website "https://www.kaggle.com/docs/api". 
4. Run the downloading script with "./data_download.sh". 

Moreover, don't forget to download the TARDIS atomic data from this address: "https://github.com/tardis-sn/tardis-refdata/raw/master/atom_data/kurucz_cd23_chianti_H_He.h5". 


## Running a Supernova Fitting

### Prepare the supernova spectral time sequence

The supernova spectra for AIAI prediction should be: 

1. Corrected for telluric lines. 
2. Removed the host galaxy light. 
3. In the unit of erg/s/Angstrom or some equivalent, not erg/s/Hz. 
4. Do not correct the redshift, milky way dust extinction or host galaxy dust extinction, because the AIAI code will do these corrections.  
5. Has two columns, the first column is the wavelength in the unit of Angstrom, the second column is the flux value. 

All the spectra should be stored in a folder named by the supernova name. For example, a series of spectra from SN2011fe are stored in the "ObserveSpectra" folder. 

Moreover, there should be a file called "starTableDen.csv" in the suparnova name folder. 
In the "starTableDen.csv" file, there are three columns for the observation phases, the file name of the spectra, and the observing telescopes. 
An example is also given in the "ObserveSpectra/SN2011fe" folder.  

### Predict the supernova element abundance

The neural network prediction function is: 

'''
from AIAISN import predictFunc
predictFunc.readNetPredictSave(snName='SN2011fe',specDir='ObserveExample/',predOutDir='predOutExample/',networkDir='../2_Train/MdSaver/110KLogML/',ebvHost=0,ebvMw=0.0088248,redshift=0.000804)
'''

"snName" is the name of the supernova, and should be matched to the folder name that stores the spectra of this supernova.  
"specDir" is the folder name that stores the spectra.  
"predOutDir" is the directory that stores the output element abundance, resampled observation spectra and other TARDIS running parameters.  
"networkDir" is the directory that stores the network weight parameters and network training histories.  
"ebvHost" is the host galaxy dust extinction value E(B-V).  
"ebvMw" is the milky way dust extinction value E(B-V).  
"redshift" is the redshift of the supernova.  

When the working directory is "5_AIAISN_Package", and the dependent packages are properly installed, and the neural network weight parameters are downloaded to the correct directory, this example prediction function should be readily executable.  

### Fit the spectra for a density

The radiative transfer spectral fitting function is: 

'''
from AIAISN import fittingFunc
fittingFunc.fitRunner(snname='SN2011fe',predOutDir='predOutExample/',specoutdir='specOutExample/',atomData='/[path_to]/kurucz_cd23_chianti_H_He.h5',cachedir='Cache/',mpiCommandor='',threadCount=8)
'''

"snname" is the name of the supernova.  
"predOutDir" is the directory that stores the otput element abundance, resampled observation spectra and other TARDIS running parameters, it should be the same as in the prediction function.  
"specoutdir" is the directory that stores the TARDIS fitting spectral time sequence, which is the major output of this function.  
"atomData" is the location of the atomic spectral line data. This atomic data file is used by TARDIS simulation, please find the file path where the atomic data is downloaded.  
"cachedir" is the directory that stores the temporary files in TARDIS simulation.  
"mpiCommandor" is the command that enables the parallel execution of TARDIS.  
"threadCount" is the number of threads used for a single TARDIS simulation task.  

#### About MPI

The spectral fitting process usually requires to run TARDIS simulation several hundreds of times, depending on the number of observed spectra. 
Therefore, to run multiple TARDIS simulation programs on different nodes in a supercomputer, some MPI functions or resource management functions are needed. 
For example, when running this fitting function on Texas A&M University HPRC, the "mpiCommandor" can be set as "tamulauncher --norelease-resources --commands-pernode 2 " . 
Please refer to your supercomputer user manual to configure the "mpiCommandor" input.  








python Predictor.py SN2011fe ObserveExample/ predOutExample/ /home/gesa/SuperCode/GithubUpload/2_Train/MdSaver/110KLogML/ 0 0.00882 0.000804
python Fitting.py SN2011fe predOutExample/ specOutExample/ kurucz_cd23_chianti_H_He.h5


