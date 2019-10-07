# Fake-News

This work was done in short duration  on Liar Plus dataset. All instructions to run the code is given below:

Python Version >=3.5

libraries required :pandas,sklearn, nltk,pytorch,torchtext

install spacy via--> -pip install spacy && python -m spacy download en

download Glove 300d embeddings->http://nlp.stanford.edu/data/glove.840B.300d.zip\\
download Glove 100d embeddings->http://nlp.stanford.edu/data/glove.6B.zip

*** TO CREATE MODIFIED DATA ****

OPEN new.py

set values of variables as described below:
train_data-> path to LIAR_PLUS train2.csv
val_data-> path to LIAR_PLUS val2.csv
train_data-> path to LIAR_PLUS test2.csv

{{** Two statements with index 2142,9375 did not had values corressponding to barely true counts, \
false counts , half true counts, mostly true counts, pants of fire counts so they have been dropped**}}



***FOR MULTICLASS CLASSIFICATION***

embed_path= Path to 100d embedding file
train_path= path to csv file after running new.py
val_path= path to csv file after running new.py
test_path= path to csv file after running new.py

Go to folder SMJ(Multi): python multi_train.py --embed_path --train_path --val_path --test_path  ## SHOULD GET A VALUE 31.6%


***FOR BINARY CLASSIFICATION***

train_path= path to csv file after running new.py
val_path= path to csv file after running new.py
test_path= path to csv file after running new.py

Go to folder SMJ(Binary):

if you want to use CNN Model->

embed_path= path to 300d glove word embeddings

python train.py --embed_path --train_path --val_path --test_path ## SHOULD GET A VALUE AROUND 64.9%



if you want to use Logisitic Regression Model ->

embed_path= Path to 100d embedding file

python binary_train.py --embed_path --train_path --val_path --test_path ## SHOULD GET A VALUE  64.44%




******MY MACHINE CONFIGURATION *********

Ram: 8GB
GPU: Nvidia GeForce 1050Ti (4GB)

