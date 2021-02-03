import librosa
import pandas as pd
import numpy as np


def extract_features(metadata_df, force_new=False):
    """
    Extract features from audio files using a mel spectrogram representation.

    :param metadata_df: DataFrame
        Pandas DataFrame consisting of metadata of a set of audio files with columns
        {'author', 'gender', 'emotion', 'file_path'}
    :param force_new: Boolean
        Indicates whether a new dataset should be created (default is False)
    :return: DataFrame
        A Pandas DataFrame consisting of the original metadata with the extracted features is returned
    """

    # Check whether metadata_df is of correct format
    if list(metadata_df.columns) != ['gender', 'emotion', 'actor', 'file_path']:
        raise Exception("Column values are not equal to ['gender', 'emotion', 'actor', 'file_path']")

    # Iterate over all audio files and extract mean values of the log Mel spectrogram
    mel_spectrogram = []
    for index, path in enumerate(metadata_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

        # get the mel-scaled spectrogram (transform both the y-axis (frequency) to log scale,
        # and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=16, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        #  Average spectrogram temporally
        log_spectrogram = np.mean(db_spec, axis=0)
        mel_spectrogram.append(log_spectrogram)

    df_combined = pd.concat([metadata_df, pd.DataFrame(mel_spectrogram)], axis=1)
    df_combined = df_combined.fillna(0)
    df_combined.drop(columns='path', inplace=True)

    return df_combined

