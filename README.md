# The code repository for the paper "Artificial Intelligence Assisted Inversion (AIAI): Quantifying the Spectral Features of $^{56}$Ni of Type Ia Supernovae" 

This is the code repository for the paper "Artificial Intelligence Assisted Inversion (AIAI): Quantifying the Spectral Features of $^{56}$Ni of Type Ia Supernovae". The code is written in jupyter notebook, and I have added some comments to illustrate the usage of the functions. 

## To read through the program

Please go through the folders "1\_Generate/", "2\_Train", "3\_Predict", "4\_Fit". The code in these folders are used to: 

(1) Generate the supernova ejecta structure, then use TARDIS radiative transfer program to simulate a spectrum for each ejecta structure.  
(2) Train a set of neural networks that predict ejecta structure using the simulated spectra as input.  
(3) Input the observed supernova spectra into the neural network to get predicted ejecta structure.  
(4) Run the TARDIS radiative transfer simulation, to fit the simulation spectra to the observed spectra.  

## To run a fitting for a new supernova



## To access the fitting results already in the paper



## Some data are too large to store in github

You can download these data from the kaggle platform using the following link, or use the ".sh" command line script at "5\_AIAISN\_Package/data\_download.sh". 

The training and testing data are available at https://www.kaggle.com/datasets/geronimoestellarchen/dltd-data , please download them and store them into "1_Genearate/DataSet/110KRun/" folder. 


The output of the tardis simulation spectra are available at https://www.kaggle.com/datasets/geronimoestellarchen/dltd-data2 , please download them and store them into "1_Generate/ContSend/" folder. 

The neural networks are available at https://www.kaggle.com/datasets/geronimoestellarchen/dltd-network , please download them and store them into "2_Train/MdSaver/110KLogML/" folder. 






