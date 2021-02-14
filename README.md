# speech-emotion-recognition
Speech Emotion Recognition group project for a course in Machine Learning.

The code of this project is inspired by Muriel Kosak's article https://towardsdatascience.com/speech-emotion-recognition-using-ravdess-audio-dataset-ce19d162690.
Her code is replicated in the file TDS-example/tds-example.py.

This code is being modified and modularized for flexibility. 
This modularized version is able to create a dataset of metadata of audio files from the RAVDESS dataset and perform a mel spectrogram feature extraction on these audio files.
Our modified version of the code furthermore contains our own versions of the models with different sets of optimizers and plots.

A complete run of this and additional code is found in the file "Optimization/Model optimization.ipynb". This Jupyter Notebook shows blocks of code followed by its corresponding output. This makes it possible to follow the entire workflow of one run of the code from loading the data into the editor to showing output from models and graphs displaying comparisons of different models run.
