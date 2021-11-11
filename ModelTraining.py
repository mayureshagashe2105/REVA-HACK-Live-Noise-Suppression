"""This script is to preprocess the data and for feature engineering, further this data is used to train the model to
deploy it on Streamlit. The model training and preprocessing part along with the outputs is present in the ipynb
version of this script. """

# Mounting Google drive to access the data

from google.colab import drive

drive.mount('/content/gdrive')

# Importing Project Dependencies

import os
import glob
from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import style
import zipfile
import pickle

# Configuring GPU for training

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Extracting zipped files into respective folders


if not os.path.exists('/content/Noise_Suprresion'):
    with zipfile.ZipFile('/content/gdrive/MyDrive/Noise supression Dataset/DS_10283_1942.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Noise_Suprresion/unzipped_dataset')

    with zipfile.ZipFile('/content/Noise_Suprresion/unzipped_dataset/clean_trainset_wav.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Noise_Suprresion/train/clean')

    with zipfile.ZipFile('/content/Noise_Suprresion/unzipped_dataset/noisy_trainset_wav.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Noise_Suprresion/train/noisy')

    with zipfile.ZipFile('/content/Noise_Suprresion/unzipped_dataset/clean_testset_wav.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Noise_Suprresion/test/clean')

    with zipfile.ZipFile('/content/Noise_Suprresion/unzipped_dataset/noisy_testset_wav.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Noise_Suprresion/test/noisy')

    print('Files unzipped successfully!')

else:
    print('Files already exists!')

# Getting list of available files in the target directories
"""This is similar to os.listdir()"""
clean_train_list = glob.glob('/content/Noise_Suprresion/train/clean/*')
noisy_train_list = glob.glob('/content/Noise_Suprresion/train/noisy/*')

"""Reading in all the files and concatenating them on top of each other to make 1 long sample of audio file"""

clean_samples, _ = tf.audio.decode_wav(tf.io.read_file(clean_train_list[0]), desired_channels=1)  # Reading clean
# audio samples
for audio in tqdm(clean_train_list[1:]):
    aud, _ = tf.audio.decode_wav(tf.io.read_file(audio), desired_channels=1)
    clean_samples = tf.concat([clean_samples, aud], axis=0)

noisy_samples, _ = tf.audio.decode_wav(tf.io.read_file(noisy_train_list[0]), desired_channels=1)  # Reading noisy
# audio samples
for audio in tqdm(noisy_train_list[1:]):
    aud, _ = tf.audio.decode_wav(tf.io.read_file(audio), desired_channels=1)
    noisy_samples = tf.concat([noisy_samples, aud], axis=0)

"""After this step, all the preprocessed data is dumped into a pickle file for effective memory utilization"""
with open('clean_samples.pkl', 'wb') as f:
    pickle.dump(clean_samples, f)
with open('noisy_samples.pkl', 'wb') as f:
    pickle.dump(noisy_samples, f)

"""Now we can reset the local runtime to free up variable RAM and then can just load these pickle files. This step is 
crucial is your GPU has a limited VRAM as pre-processed tensors are of size > 13GB"""

# Loading in the pickle files

pickle_in = open('clean_samples.pkl', 'rb')
clean_samples = pickle.load(pickle_in)
clean_samples = tf.cast(clean_samples, 'float16')

pickle_in = open('noisy_samples.pkl', 'rb')
noisy_samples = pickle.load(pickle_in)
noisy_samples = tf.cast(noisy_samples, 'float16')

# Sampling the tensors with size 16000 and stacking them on top of together

clean_train, noisy_train = [], []
sampling_size = 16000
for i in range((noisy_samples.shape[0]) // sampling_size):
    clean_train.append(clean_samples[i * sampling_size:i * sampling_size + sampling_size])
    noisy_train.append(noisy_samples[i * sampling_size:i * sampling_size + sampling_size])

noisy_train = tf.stack(noisy_train)
clean_train = tf.stack(clean_train)

"""Here the sampling size is decided arbitrarily. This is to ensure fast training with incurring minimum loss. This 
sample size will not matter any of the future training part as we will be training our model with input shape as (
None, 1) """

# Data Visualization

"""Visualizing our training set's random noisy audio sample and it's counterpart"""

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

axes[0][0].plot(np.arange(noisy_train[10].shape[0]), noisy_train[10], color='r')
axes[1][0].plot(np.arange(clean_train[10].shape[0]), clean_train[10], color='b')

axes[0][1].plot(np.arange(noisy_train[100].shape[0]), noisy_train[100], color='r')
axes[1][1].plot(np.arange(clean_train[100].shape[0]), clean_train[100], color='b')

plt.show()


# Creating Input pipeline for effective training and to avoid data loading overhead

def input_pipeline(X, y, batch_size, shuffle_buffer, train_split=0.8):
    """
    This function will create a data pipeline to avoid data loading overhead. It will return 2 datasets of
    <class: tf.prefetched_data>. This is to ensure effective and minimum training time

    :param X: Noisy audio samples
    :param y: Clean audio samples as counterpart of X
    :param batch_size: batch_size for training and validation
    :param shuffle_buffer: initializing a shuffle_buffer for fast shuffling
    :param train_split: percentage of data to be used while training (train_test_split)
    :return train_ds.prefetch(1), val_ds.prefetch(1): training and validation datasets (tf.prefetched_data)
    """
    train_split_index = int(X.shape[0] * 0.8)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(shuffle_buffer)
    train_ds, val_ds = ds.take(train_split_index), ds.skip(train_split_index)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds.prefetch(1), val_ds.prefetch(1)


train_ds, val_ds = input_pipeline(noisy_train, clean_train, 64, 50000)
