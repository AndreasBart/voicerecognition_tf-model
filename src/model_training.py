import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import analyze_data
import load_data

class model:
    def __init__(self, data, analyze):
        
        self.data = data
        self.analyze = analyze

        train_ds = spectrogram_ds
        val_ds = preprocess_dataset(val_files)
        test_ds = preprocess_dataset(test_files)
        batch_size = 64
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)


    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=self.analyze.get_waveform_and_label,
            num_parallel_calls=self.analyze.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=self.analyze.get_spectrogram_and_label_id,
            num_parallel_calls=self.analyze.AUTOTUNE)
        return output_ds