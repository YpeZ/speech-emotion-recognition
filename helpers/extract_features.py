import librosa
import pandas as pd
import numpy as np
import os


def extract_features(metadata_df=None, set_name='RAVDESS', force_new=False):
    """
    Extract features from audio files using a mel spectrogram representation.


    :param metadata_df: DataFrame
        Pandas DataFrame consisting of metadata of a set of audio files with columns
        {'author', 'gender', 'emotion', 'file_path'} (default is None)
    :param set_name: str
        The name of the metadata set to be used
    :param force_new: Boolean
        Indicates whether a new dataset should be created (default is False)
    :return: DataFrame
        A Pandas DataFrame consisting of the original metadata with the extracted features is returned
        (default is False)
    """

    print(f"Using dataset {set_name}")

    # Check whether metadata_df is of correct format
    if list(metadata_df.columns) != ['gender', 'emotion', 'actor', 'file_path']:
        raise Exception("Column values are not equal to ['gender', 'emotion', 'actor', 'file_path']")

    filepath = os.path.dirname(__file__)
    if not force_new:
        try:
            df_combined = pd.read_csv(os.path.join(filepath, '../Data/') + set_name + '_features.csv')

            return df_combined
        except FileNotFoundError:
            # If creation of a new file is not forced but an old file is not found, create a new one anyway
            pass

    print("Creating new feature dataset")

    # Iterate over all audio files and extract mean values of the log Mel spectrogram
    mel_spectrogram = []
    for index, path in enumerate(metadata_df.file_path):
        x, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

        # get the mel-scaled spectrogram (transform both the y-axis (frequency) to log scale,
        # and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes)
        spectrogram = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=16, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        #  Average spectrogram temporally
        log_spectrogram = np.mean(db_spec, axis=0)
        mel_spectrogram.append(log_spectrogram)

    df_combined = pd.concat([metadata_df, pd.DataFrame(mel_spectrogram)], axis=1)
    df_combined = df_combined.fillna(0)
    df_combined.drop(columns='file_path', inplace=True)

    df_combined.to_csv(os.path.join(filepath, '../Data/') + set_name + '_features.csv',
                       index=False)
    return df_combined
