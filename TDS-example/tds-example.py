# Import libraries
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(sys.path, end="\n")

dir_path = sys.path[-1]+"/TDS-example/"
print(dir_path)

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tikzplotlib
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical

from plotters import confusion_plot

### Create waveplot and mel spectrogram of neutral male voice
filename = 'RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav'
x, sr = librosa.load(filename)

plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sr)
plt.title('Waveplot - Male Neutral')
plt.savefig(dir_path+'Waveplot_MaleNeutral.png')
# tikzplotlib.save('Waveplot_MaleNeutral.tex')

# Create log Mel spectogram
spectogram = librosa.feature.melspectrogram(y=x, sr=sr,
                                            n_mels=128, fmax=800)
spectogram = librosa.power_to_db(spectogram)

librosa.display.specshow(spectogram, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectogram - Male Neutral')
plt.colorbar(format='%+2.0f dB')
plt.savefig(dir_path+'MelSpec_Male_Neutral.png')
plt.interactive(False)
# tikzplotlib.save('melspec_male_neutral.tex')
plt.clf()

### Create waveplot and mel spectrogram of male angry voice
filename = 'RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-05-01-01-01-01.wav'
x, sr = librosa.load(filename)

# Create waveplot
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sr)
plt.title('Waveplot - Male Angry')
plt.savefig(dir_path+'Waveplot_MaleAngry.png')
# tikzplotlib.save('Waveplot_MaleAngry.tex')


# Create log Mel spectogram
spectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=800)
spectogram = librosa.power_to_db(spectogram)

librosa.display.specshow(spectogram, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectogram - Male Angry')
plt.colorbar(format='%+2.0f dB')
plt.savefig(dir_path+'MelSpec_Male_Angry.png')
plt.interactive(False)
plt.clf()


### Locate audio files
audio = 'RAVDESS/Audio_Speech_Actors_01-24/'
actor_folders = os.listdir(audio)  # list files in audio directory
actor_folders.sort()

### Prepare the data
# Create empty lists for emotion, gender, actor and file path data
emotion = []
gender = []
actor = []
file_path = []
for i in actor_folders:
    filename = os.listdir(audio + i)            # iterate over Actor folders
    for f in filename:                          # go through files in Actor folder
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        actor.append(int(part[6]))
        bg = int(part[6])
        if bg % 2 == 0:
            bg = "female"
        else:
            bg = "male"
        gender.append(bg)
        file_path.append(audio + i + '/' + f)

# Join the lists into a dataframe
audio_df = pd.DataFrame(data={'gender': gender, 'emotion': emotion, 'actor': actor, 'path': file_path})

audio_df.emotion = audio_df.emotion.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                                             5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})

# Inspect a sample of the new dataset
print(audio_df.sample(10))
# Look at distribution of classes
print(audio_df.emotion.value_counts())

# Export df to csv
audio_df.to_csv(dir_path+'Output/audio.csv')

# Iterate over all audio files and extract mean values of the log Mel spectrogram
mel_spectrogram = []
for index, path in enumerate(audio_df.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

    # get the mel-scaled spectrogram (transform both the y-axis (frequency) to log scale,
    # and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=16, fmax=8000)
    db_spec = librosa.power_to_db(spectrogram)
    #  Average spectrogram temporally
    log_spectrogram = np.mean(db_spec, axis=0)
    mel_spectrogram.append(log_spectrogram)

df_combined = pd.concat([audio_df, pd.DataFrame(mel_spectrogram)], axis=1)
df_combined = df_combined.fillna(0)
df_combined.drop(columns='path', inplace=True)
print(df_combined.head())

### Data Preprocessing

# Split data into train/test
train, test = train_test_split(df_combined, test_size=0.2,
                               random_state=0,
                               stratify=df_combined[['emotion', 'gender', 'actor']])

X_train = train.iloc[:, 3:]
y_train = train.iloc[:, :2].drop(columns=['gender'])
print(X_train.shape)
X_test = test.iloc[:, 3:]
y_test = test.iloc[:, :2].drop(columns=['gender'])
print(X_test.shape)

# Baseline model
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='stratified')
pred = dummy_clf.predict(X_test)
dummy_score = dummy_clf.score(X_test, y_test)
cm = confusion_matrix(y_test, pred)


confusion_plot(y_test, pred, name=dir_path+"Dummy Classifier Model")


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
clf_score = clf.score(X_test, y_test)

# plot_confusion_matrix(clf, X_test, y_test)
# plt.savefig('dectree_conf_mat.png')

confusion_plot(y_test, pred, name=dir_path+"Decision Tree Model")

# Normalize the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Transform into Keras
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# One-hot encode variables
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
print(lb.classes_)

# Reshape to include 3D tensor
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]


### Build 1D CNN layers
model = tf.keras.Sequential()
model.add(Conv1D(64, kernel_size=20, activation='relu', input_shape=(X_train.shape[1], 1)))

model.add(Conv1D(128, kernel_size=20, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=8))
model.add(Dropout(0.2))

model.add(Conv1D(128, kernel_size=20, activation='relu'))
model.add(MaxPooling1D(pool_size=8))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))
print(model.summary())
opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)

checkpoint = ModelCheckpoint(dir_path+'face_detector.h15', monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1)

# FIT MODEL
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model_history = model.fit(X_train, y_train, batch_size=8, epochs=100,
                          validation_data=(X_test, y_test), callbacks=checkpoint)

model.save_weights(dir_path+"best_model_weights.h5")
model.save(dir_path+'best_model.h5')
print("Saved model to disk")

cnn_score = model.evaluate(X_test, y_test)[1]

pred = model.predict(X_test, batch_size=32)
pred = pred.argmax(axis=1)
pred = pred.astype(int).flatten()
pred = (lb.inverse_transform(pred))

actual = y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform(actual))

confusion_plot(actual, pred, dir_path+'CNN Model')

# Compare performance between models
print(f"Dummy score: '{round(dummy_score, 3)}'")
print(f"Dtree score: '{round(clf_score, 3)}'")
print(f"CNN score:   '{round(cnn_score, 3)}'")


# Plot
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(dir_path+'Augmented_Model_Accuracy.png')
tikzplotlib.save(dir_path+'augmented_model_accuracy.tex')
plt.clf()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('CNN Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(dir_path+'Augmented_Model_Loss.png')
tikzplotlib.save(dir_path+'augmented_model_loss.tex')
plt.clf()
