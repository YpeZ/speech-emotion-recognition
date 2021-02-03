from helpers import *

import os
import librosa

### Load dataset
dataset = create_audio_file_df(set_name='RAVDESS', force_new=True)

print(dataset.sample(10))

### Feature extraction using Mel Spectrogram
dataset = extract_features(dataset, force_new=True)

### Data preprocessing


### Create CNN Model


### Inspect results
