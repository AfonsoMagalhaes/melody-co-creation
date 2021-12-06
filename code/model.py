#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:28:30 2021

@author: afonsomagalhaes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from preprocessor import generate_training_sequences, load_mapping, SEQUENCE_LENGTH

def get_n_vocab():  
    """Get number of symbols in the dataset vocabulary.
    :return length of vocabulary (int):
    """
    
    return len(load_mapping())

def get_best_weights_file():
    """Get filename of the file with the weights for the smallest loss value of training.
    :return best_weights (str):
    """
    
    filenames = os.listdir("../weights/")
    
    min_loss = 1000
    for filename in filenames:
        if filename[-4:] == "hdf5":
            loss = filename.split("-")[3]
            
            if float(loss) < min_loss:
                min_loss = float(loss)
                best_weights = filename
    
    return best_weights

def build_model(n_vocab, weights=None):
    """Build the neural network to train and load the weights from training, if needed.
    :param n_vocab (int): Number of symbols in the dataset vocabulary.
    :param weights (str): "best" to load the weights with the least loss value, "last" to load the last epochs weights.
    :return model (keras Sequential model):
    """
    
    model = keras.Sequential()
    model.add(layers.LSTM(100, input_shape=(None, 1), return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(100))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_vocab))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    if len(os.listdir("../weights/")) == 0:
        return
        
    if weights == "best":
        model.load_weights("../weights/" + get_best_weights_file())
    elif weights == "last":
        model.load_weights("../weights/" + os.listdir("../weights/")[-1])
        
    return model

def train(model, network_input, network_target):
    """Train the neural network and save it's weights each time the loss value improves.
    :param model (keras Sequential model): Neural network to train.
    :param network_input (list(int)): Training input divided in sequences of size sequence length.
    :param network_target (list): One-hot encoded training target.
    """
    
    filepath = "../weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    
    callbacks_list = [checkpoint]
    
    model.fit(network_input, network_target, epochs=200, batch_size=64, callbacks=callbacks_list)

def plot_loss():
    """Plot training loss value across epochs, from the weights files saved."""
    
    epochs_loss = []
    
    for (dirpath, dirnames, filenames) in os.walk("../weights/"):
        for filename in filenames:
            if filename[-4:] == "hdf5":
                epoch, loss = filename.split("-")[2:4]
                epochs_loss.append([int(epoch), float(loss)])
    
    epochs_loss = np.array(epochs_loss)
    epochs_loss = epochs_loss[epochs_loss[:, 0].argsort()]
    
    plt.figure()
    plt.plot(epochs_loss[:,0], epochs_loss[:,1])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
            
if __name__ == "__main__":
    #network_input, network_target = generate_training_sequences(SEQUENCE_LENGTH)

    #model = build_model(get_n_vocab(), weights="last")
    #train(network_input, network_target)
    
    plot_loss()
    
    