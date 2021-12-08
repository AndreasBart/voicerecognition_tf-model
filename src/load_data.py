import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


class Data:
      def __init__(self):
            self.train_files = None
            self.val_files = None
            self.test_files = None
            self.commands = None
            


      def getData(self):
            # Set seed for experiment reproducibility
            seed = 42
            tf.random.set_seed(seed)
            np.random.seed(seed)

            # Get the data
            data_dir = pathlib.Path('data/mini_speech_commands')
            if not data_dir.exists():
                  tf.keras.utils.get_file(
                  'mini_speech_commands.zip',
                  origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                  extract=True,
                  cache_dir='.', cache_subdir='data')

            #check basic statistics
            self.commands = np.array(tf.io.gfile.listdir(str(data_dir)))
            self.commands = self.commands[self.commands != 'README.md']
            print('Commands:', self.commands)

            #Extract the audio files into a list and shuffle it.
            filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
            filenames = tf.random.shuffle(filenames)
            num_samples = len(filenames)
            print('Number of total examples:', num_samples)
            print('Number of examples per label:', len(tf.io.gfile.listdir(str(data_dir/self.commands[0]))))
            #print('Example file tensor:', filenames[0])

            #Split the files into training, validation and test sets using a 80:10:10 ratio, respectively.
            self.train_files = filenames[:6400]
            self.val_files = filenames[6400: 6400 + 800]
            self.test_files = filenames[-800:]

      def size(self):
            print('Training set size', len(self.train_files))
            print('Validation set size', len(self.val_files))
            print('Test set size', len(self.test_files))
