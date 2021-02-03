import os
import pandas as pd


def create_audio_file_df(set_name='RAVDESS', force_new=False) -> pd.DataFrame:
    """
    Create or retrieve a data table of metadata for each audio file, i.e. speaker id, gender, emotion and file path and
    write it to csv.

    Parameters
    ----------
    set_name : str
        The name of the dataset used (default is RAVDESS)
    force_new : Boolean
        Indicates whether a new dataset should be created (default is False)
    """

    filepath = os.path.dirname(__file__)
    inside_folder = ''
    if set_name == 'RAVDESS':
        inside_folder = 'Audio_Speech_Actors_01-24/'

    if not force_new:
        try:
            audio_df = pd.read_csv(os.path.join(filepath, '../Data/') + set_name + '_list.csv')

            return audio_df
        except FileNotFoundError:
            # If creation of a new file is not forced but an old file is not found, create a new one anyway
            pass

    audio_folder = os.path.join(filepath, '/'.join(['..', set_name, inside_folder]))
    actor_folders = os.listdir(audio_folder)
    actor_folders.sort()

    # Prepare data as list
    dictionary_list = []
    # Go through all folders with audio files
    for i in actor_folders:
        # Go through all files in folder
        filename = os.listdir(audio_folder + i)
        for f in filename:
            part = f.split('.')[0].split('-')
            emotion = int(part[2])
            actor = int(part[6])
            if actor % 2 == 0:
                gender = "female"
            else:
                gender = "male"
            file_path = audio_folder + i + '/' + f

            # Append new data to the dictionary list
            dictionary_data = {'gender': gender, 'emotion': emotion, 'actor': actor, 'file_path': file_path}
            dictionary_list.append(dictionary_data)

    # Create DataFrame from dictionary list
    audio_df = pd.DataFrame(dictionary_list)
    # Name the emotion attributes
    audio_df.emotion = audio_df.emotion.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                                                 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})

    # Write dataset to csv
    audio_df.to_csv(os.path.join(filepath, '../Data/') + set_name + '_list.csv',
                    index=False)

    return audio_df


if __name__ == '__main__':
    data = create_audio_file_df(force_new=True)
    print(data.sample(10))
