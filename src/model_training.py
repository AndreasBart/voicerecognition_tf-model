import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tempfile

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from tensorflow.keras.layers.experimental import preprocessing
import tensorflowjs as tfjs
import analyze_data
import load_data

class Model:
    def __init__(self, data, analyze):
        
        self.data = data
        self.analyze = analyze
        self.tfjs_target_dir = "C:/Users/baa37164/Desktop/model"

        self.train_ds = self.analyze.spectrogram_ds
        self.val_ds = self.preprocess_dataset(self.data.val_files)
        self.test_ds = self.preprocess_dataset(self.data.test_files)

        batch_size = 64
        self.train_ds = self.train_ds.batch(batch_size)
        self.val_ds = self.val_ds.batch(batch_size)

        self.train_ds = self.train_ds.cache().prefetch(self.analyze.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(self.analyze.AUTOTUNE)
        self.model = models.Sequential


    def preprocess_dataset(self, files):

        files_ds = tf.data.Dataset.from_tensor_slices(files)            
        output_ds = files_ds.map(
        map_func=self.analyze.get_waveform_and_label,
        num_parallel_calls = self.analyze.AUTOTUNE)
        output_ds = output_ds.map(
        map_func=self.analyze.get_spectrogram_and_label_id,
        num_parallel_calls=self.analyze.AUTOTUNE)
        return output_ds

    def build_train_model(self):
        for spectrogram, _ in self.analyze.spectrogram_ds.take(1):
            input_shape = spectrogram.shape
        print('Input shape:', input_shape)
        num_labels = len(self.data.commands)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms with `Normalization.adapt`.
        norm_layer.adapt(data=self.analyze.spectrogram_ds.map(map_func=lambda spec, label: spec))

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        #100 Epochs to allow the model to train until early stop
        EPOCHS = 100
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        tfjs.converters.save_keras_model(self.model, self.tfjs_target_dir)
        print(pathlib.Path().resolve())

        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.show()

    #Test accuracy for all labels
    def confusion_matrix(self):
        test_audio = []
        test_labels = []

        for audio, label in self.test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        y_pred = np.argmax(self.model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')
    
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=self.data.commands, yticklabels=self.data.commands, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()


    #Test own files  
    def inference(self):
        #Path to the tested audiofile
        sample_file = '../data/mini_speech_commands/no/1b4c9b89_nohash_3.wav'

        sample_ds = self.preprocess_dataset([str(sample_file)])

        for spectrogram, label in sample_ds.batch(1):
            prediction = self.model(spectrogram)
            plt.bar(self.data.commands, tf.nn.softmax(prediction[0]))
            plt.title(f'Predictions for "{self.data.commands[label[0]]}"')
            plt.show()

    #Save model as SavedModel format
    def saveModel(self):
        
        MODEL_DIR = tempfile.gettempdir()
        version = 1
        export_path = os.path.join(MODEL_DIR, str(version))
        print('export_path = {}\n'.format(export_path))

        tf.keras.models.save_model(
            self.model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
        print('Saved successfully')
