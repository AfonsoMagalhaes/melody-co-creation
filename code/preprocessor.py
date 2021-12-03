#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:27:48 2021

@author: afonsomagalhaes
"""

import os
import music21 as m21
import json
import numpy as np
from tensorflow import keras

SAVE_DIRECTORY = "../dataset"
SINGLE_FILE_DATASET = "../file_dataset"
SEQUENCE_LENGTH = 75
MAPPING_PATH = "../mapping.json"

def import_songs(dataset_dir, num_files_per_type=None):
    songs = []
       
    for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
        n = 0
        for filename in filenames:
            if filename[-3:] == "krn":
                songs.append(m21.converter.parse(os.path.join(dirpath, filename)))
                n += 1
                
            if n == num_files_per_type:
                break
    
    return songs

def transpose(song):
    """Transposes song to C maj/A min
    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """

    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode(song):
    encoded_song = []
    notes_to_parse = song.flat.notesAndRests

    for element in notes_to_parse:
        if isinstance(element, m21.note.Note):
            symbol = element.pitch.midi
        elif isinstance(element, m21.note.Rest):
            symbol = "r"
        
        encoded_song.append(str(symbol))
    
        steps = int(element.duration.quarterLength / 0.25) - 1
        for _ in range(steps):
            encoded_song.append("_")
    encoded_song = " ".join(map(str,encoded_song))

    return encoded_song

def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()
    return song

def create_file_dataset(dataset_path,file_dataset_path,length_sequence):
    
    new_song_delimi = "/ " * length_sequence
    songs =""
    
    #load encoded songs
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimi
    songs =songs[:-1]
    
    #save string that contain all datasets
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
        
def create_mapping(file_dataset, mapping_path):

    mappings = {}
    
    #identify vocabulary(all the simbols in dataset)
    songs = load(file_dataset)
    songs = songs.split()
    vocabulary = list(set(songs))
    
    #mapping
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i
        
    #save vocabulary in json file
    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp,indent=4)
    
def load_mapping():
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)
    
    return mapping
   

# Convert song encoded symbols to list of integers
def convert_songs_to_int(songs):
    int_songs = []

    mapping = load_mapping()
    # transform songs string to list of symbols
    songs = songs.split()

    # map symbols to numeric values
    for symbol in songs:
        int_songs.append(mapping[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.
    :param sequence_length (int): Length of each sequence. 
    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
        
    n_vocab = len(set(int_songs))

    # one-hot encode the sequences
    n_patterns = len(inputs)
    network_input = np.reshape(inputs, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    
    network_output = keras.utils.to_categorical(targets)

    return network_input, network_output

def preprocess(num_files_per_type):
    
    songs = import_songs("../deutschl", num_files_per_type)
    
    for i,song in enumerate(songs):
        song = transpose(song)
        
        encoded_song = encode(song)
        save_path = os.path.join(SAVE_DIRECTORY,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)
    
    create_file_dataset(SAVE_DIRECTORY,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(SINGLE_FILE_DATASET, MAPPING_PATH)

if __name__ == "__main__":
    preprocess(num_files_per_type=20)
    