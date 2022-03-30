# Level 4 Dissertation Project

Author: Anuraj Taya
Student ID: 2334152T

## Improving Explainability using Expected gradients for ECG classification

In this project we study various gradient based explainability methods and examine the change in model performance with use of attribution methods. 

##### Directory structure:

The following structure showcases the data structure of the project and explanations for each file.

|---Data  (Would contain the preprocess data)
|---Data_processing
|    |---data_preprocess.ipynb  (Extracting the beats)
|    |---data_split_resample.ipynb (Resampling the data for patient leave-out method)
|---ECG
|    |---eager_ops.py  (Expected gradients methods)
|    |---IG.py  (Integrated gradients methods)
|---cnn_ph_eg.py  (CNN beat hold-out with EG)
|---cnn_ph.py  (CNN beat hold-out)
|---cnn_pl_eg.py (CNN patient leave-out with EG)
|---cnn_pl.py  (CNN patient leave-out)
|---generate_teset_y.py  (Creating the beat hold-out data for graph generations)
|---generating_eg_att.py  (Generates the Expected gradient attributions for every model)
|---generating_ig_att.py  (Generates the Integrated gradient attributions for every model)
|---graphs.ipynb  (Used to create the graphs for the thesis)
|---lstm_ph_eg.py  (LSTM beat hold-out with EG)
|---lstm_ph.py  (LSTM beat hold-out)
|---lstm_pl_eg.py  (LSTM patient leave-out with EG)
|---lstm_pl.py  (LSTM patient leave-out)
|---stats.ipynb  (Used to create the statical diagrams and tables)
|---svc_ph.py (SVC beat hold-out)
|---svc_pl.py  (SVC patient leave-out)
|---.gitignore
|---README.md
|---manual.md

All the models were run on the GPU due to the long training times.