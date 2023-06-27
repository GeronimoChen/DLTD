mkdir -p DataSet/110KRun/
cd DataSet/110KRun/
kaggle datasets download -d geronimoestellarchen/dltd-data
unzip dltd-data.zip
cd ../../
mv DataSet ../1_Generate/

mkdir ContSend
cd ContSend
kaggle datasets download -d geronimoestellarchen/dltd-data2
unzip dltd-data2.zip
cd ..
mv ContSend ../1_Generate/

mkdir -p MdSaver/110KLogML
cd MdSaver/110KLogML
kaggle datasets download -d geronimoestellarchen/dltd-network
unzip dltd-network.zip
cd ../../
mv MdSaver ../2_Train/




