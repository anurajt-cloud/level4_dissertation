# Level 4 Dissertation Project

Author: Anuraj Taya
Student ID: 2334152T

## Improving Explainability using Expected gradients for ECG classification

In this project we study various gradient based explainability methods and examine the change in model performance with use of attribution methods. 

### Abstract
<pr>
  Heart diseases resulting in heart attacks and strokes are a major cause for deaths around the
world. Studies have revealed that these attacks can be forestalled by detection of arrhythmia.
Advancements in machine learning techniques have resulted in various neural networks being
employed to successfully detect arrhythmia by classifying beats into normal and abnormal classes.
However, their use has mostly been restricted to research rather than full-scale deployment in
medicine due to the lack of explainability for the model predictions, i.e., the black box design.
  In our thesis we adopt the explainability method known as Expected Gradients, an evolution
of Integrated gradients, to provide attributions that justify model predictions. We further
incorporate the loss generated from the attributions into the training process to help improve
model performances. The attribution method is applied to a convolutional neural network (CNN)
consisting of four residual blocks with residual skip connections and an 11-layer long short-term
memory neural network (LSTM) to measure its effectiveness. Data belonging to 47 patients from
the MIT-BIH arrhythmia dataset was used to pre-process and extract heart beats. These beats
were split into two evaluation datasets using the beat hold-out and patient leave-out method.
This was done to mimic the real life scenario of models predicting seen and unseen patients’
data. 
  The results showcase that integration of the expected gradient attributions into the training
process helps improve both the models’ precision and recall on unseen patient data. These results
are further supported by the visualisations of the generated attributions and statistical tests like
Kruskal test, Wilcoxon Signed Rank test, and F-oneway ANOVA test.

</pr>

### Directory structure:

The following structure showcases the data structure of the project and explanations for each file.

<pre>
|---Data  (Would contain the preprocess data)
|---Data_processing
|    |---data_preprocess.ipynb      (Extracting the beats)
|    |---data_split_resample.ipynb  (Resampling the data for patient leave-out method)
|---ECG
|    |---eager_ops.py               (Expected gradients methods)
|    |---IG.py                      (Integrated gradients methods)
|---cnn_ph_eg.py            (CNN beat hold-out with EG)
|---cnn_ph.py               (CNN beat hold-out)
|---cnn_pl_eg.py            (CNN patient leave-out with EG)
|---cnn_pl.py               (CNN patient leave-out)
|---generate_teset_y.py     (Creating the beat hold-out data for graph generations)
|---generating_eg_att.py    (Generates the Expected gradient attributions for every model)
|---generating_ig_att.py    (Generates the Integrated gradient attributions for every model)
|---graphs.ipynb            (Used to create the graphs for the thesis)
|---lstm_ph_eg.py           (LSTM beat hold-out with EG)
|---lstm_ph.py              (LSTM beat hold-out)
|---lstm_pl_eg.py           (LSTM patient leave-out with EG)
|---lstm_pl.py              (LSTM patient leave-out)
|---stats.ipynb             (Used to create the statical diagrams and tables)
|---svc_ph.py               (SVC beat hold-out)
|---svc_pl.py               (SVC patient leave-out)
|---.gitignore
|---README.md
|---manual.md 
</pre>

All the models were run on the GPU due to the long training times.
