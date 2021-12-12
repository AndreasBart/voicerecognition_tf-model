import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from IPython import display

class analyze_data:

  def __init__(self, trainfiles, commands):

      # init vars
      self.train_files = trainfiles 
      self.commands = commands
      self.spectrogram_ds = None
      self.AUTOTUNE = tf.data.AUTOTUNE

      files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)
      waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=self.AUTOTUNE)

      #Analyze spectogramms of  
      rows = 3
      cols = 3
      n = rows*cols
      fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

      for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)

      for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        self.waveform = waveform
        spectrogram = self.get_spectrogram(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)
        print('Audio playback')
        display.display(display.Audio(waveform, rate=16000))

      fig, axes = plt.subplots(2, figsize=(12, 8))
      timescale = np.arange(waveform.shape[0])
      axes[0].plot(timescale, waveform.numpy())
      axes[0].set_title('Waveform')
      axes[0].set_xlim([0, 16000])
      self.plot_spectrogram(spectrogram.numpy(), axes[1])
      axes[1].set_title('Spectrogram')
      #plt.show()

      self.spectrogram_ds = waveform_ds.map(
        self.get_spectrogram_and_label_id, num_parallel_calls=self.AUTOTUNE)
      rows = 3
      cols = 3
      n = rows*cols
      fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
      for i, (spectrogram, label_id) in enumerate(self.spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        self.plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')
      #plt.show()

  def decode_audio(self, audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

  def get_label(self, file_path):
    parts = tf.strings.split(input = file_path, sep = os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-2]

  def get_waveform_and_label(self, file_path):
    label = self.get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = self.decode_audio(audio_binary)
    return waveform, label

  #plt.show()

  def get_spectrogram(self, waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

  def plot_spectrogram(self, spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
    log_spec = np.log(spectrogram.T+np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

  def get_spectrogram_and_label_id(self, audio, label):
    spectrogram = self.get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == self.commands)
    return spectrogram, label_id




  


  

  