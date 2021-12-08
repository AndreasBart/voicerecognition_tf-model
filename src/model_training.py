import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from tensorflow.keras.layers.experimental import preprocessing
import analyze_data
import load_data

class Model:
    def __init__(self, data, analyze):
        
        self.data = data
        self.analyze = analyze

        self.train_ds = analyze.spectrogram_ds
        self.val_ds = self.preprocess_dataset(data.val_files)
        self.test_ds = self.preprocess_dataset(data.test_files)
        batch_size = 64
        train_ds = self.train_ds.batch(batch_size)
        val_ds = self.val_ds.batch(batch_size)
        train_ds = train_ds.cache().prefetch(analyze.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(analyze.AUTOTUNE)


    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=self.analyze.get_waveform_and_label,
            num_parallel_calls=self.analyze.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=self.analyze.get_spectrogram_and_label_id,
            num_parallel_calls=self.analyze.AUTOTUNE)
        return output_ds


    #Modell bauen und trainieren

    def build_train_model(self):
        for spectrogram, _ in self.train_ds.take(1):
            input_shape = spectrogram.shape
        print('Input shape:', input_shape)
        num_labels = len(self.data.commands)

        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(self.analyze.spectrogram_ds.map(lambda x, _: x))

        model = models.Sequential([
            layers.Input(shape=input_shape),
            preprocessing.Resizing(32, 32), 
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

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        EPOCHS = 10
        history = model.fit(
            self.train_ds, 
            validation_data=self.val_ds,  
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.show()


    #Leistung des Testsatzes bewerten
    def confusion_matrix(self):
        test_audio = []
        test_labels = []

        for audio, label in self.test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        y_pred = np.argmax(Model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')
    
    #Zeigt Verwirrungsmatrix an
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=self.data.commands, yticklabels=self.data.commands, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()


    #Inferenz für eine Audiodatei ausführen   
    def inferenz(self):
        sample_file = '..\data\mini_speech_commands\no\0ab3b47d_nohash_0.wav'

        sample_ds = self.preprocess_dataset([str(sample_file)])

        for spectrogram, label in sample_ds.batch(1):
            prediction = Model(spectrogram)
            plt.bar(self.data.commands, tf.nn.softmax(prediction[0]))
            plt.title(f'Predictions for "{self.data.commands[label[0]]}"')
            plt.show()
