# Instructions for running the code

All models would use a GPU if they are available. 

### Package requirements
- Python 3.8.10
- Numpy 1.22.3
- Tensorflow 2.7.1
- WFDB 3.4.1
- Pandas 1.4.1
- Scikit-learn 1.0.2
- Scipy 1.8.0
- Glob installed with python
- Matplotlib 3.5.1
- Seaborn 0.11.2

### Folder requirements

- Data - Data folder should exist in the home directory and it should contain the .csv data files. README.md contains the directory structure.
- eval_data_10k - The folder would contain the generated performance metrics and the models.

### Command to run the models

The models can be run using the following commands:
- Without GPU acceleration
*python python_file_name*
- With GPU acceleration
*python CUDA_VISIBLE_DEVICES=gpu_number python_file_name*
