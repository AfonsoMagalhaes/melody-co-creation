#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:27:55 2021

@author: afonsomagalhaes
"""

import os
import numpy as np
import random
import music21 as m21
from preprocessor import convert_songs_to_int, encode, load_mapping, SAVE_DIRECTORY
from model import build_model, get_n_vocab
from evaluator import Evaluator, INPUT_DIR

SEED_MELODY = "../melody/seed_melody.mid"
PREDICTED_MELODY = "../melody/predicted_melody"
FINAL_MELODY = "../melody/final_melody"

class MelodyGenerator:
    """Class that uses the trained model to generate melodies with or without intervention of the user."""

    def __init__(self, seed_melody=None, seed_num_notes=None, seed_pos="random",temperature=None, num_iterations=1, correction=False):
        """Constructor that initializes the Melody Generator.
        :param seed_melody (str): file path of the seed melody.
        :param start_num_notes (int): number of notes required in the seed melody (note count).
        :param temperature (float): Value from 0 to 1 that determines the determinism of the model. 
                                    Higher value, less determinism.
        :param num_iterations (int): number of (co-)generation iterations.
        :param correction (bool): True if the user can correct the generated melody in each iteration, 
                                  False otherwise.
        """
        
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.correction = correction
        self.symbols = list(load_mapping().keys())
        self.seed_num_notes = seed_num_notes
        
        # get the model with the best weights (that offered the less loss value in training)
        self.model = build_model(get_n_vocab(), weights="best")

        # get encoded seed melody and save it
        if seed_melody == None:
            self.seed_melody = self.get_seed_mel(seed_pos)
        else:
            self.seed_melody = self.get_seed_mel(seed_pos, seed_melody)
        
        self.create_midi_file(self.seed_melody, SEED_MELODY)

    def get_seed_mel(self, seed_pos, filename=None):
        """Get the seed melody for the (co-)generation.
        :param filename (str): file path of the input melody. If None, get a random file from the encoded dataset.
        :return seed_melody (str): 
        """

        # get melody string. If filename is None, get a random song from the encoded dataset.
        if filename == None:
            music_file = random.choice(os.listdir(SAVE_DIRECTORY))
            with open(SAVE_DIRECTORY + music_file,"r") as fp:
                melody_string = fp.read()
        else:
            melody_string = encode(m21.converter.parse(INPUT_DIR + filename))

        melody_string = melody_string.split()
        melody_pitches = list(map(str, filter(lambda symbol: symbol != '_' and symbol != "r", melody_string)))

        # if the start melody length is undefined or bigger than the whole melody, update it an return the whole melody
        if self.seed_num_notes == None or self.seed_num_notes > len(melody_pitches):
            self.seed_num_notes = len(melody_pitches)
            return " ".join(melody_string)

        # check f the seed is supposed to be fetched from the start or a random position
        if seed_pos == "random":
          # while loop to get correct melody
          melody_found = False
          while melody_found == False:
              
              # get random index in the melody
              random_index = int(random.random() * (len(melody_string) - self.seed_num_notes))
              
              # check if the beginning of input_melody is a pitch
              if (melody_string[random_index] == "_") or (melody_string[random_index] == "r"):
                  continue
              
              num_notes = 1
              melody_substring = [melody_string[random_index]]
              for i in range(random_index, len(melody_string)):
                  if num_notes == self.seed_num_notes:
                      break
                  
                  melody_substring.append(melody_string[i])
                  if melody_string[i] != "_" and melody_string[i] != "r":
                      num_notes += 1
              
              if num_notes < self.seed_num_notes:
                  continue
              
              melody_found = True

        elif seed_pos == "start":
          
          num_notes = 0
          melody_substring = []
          for i in range(len(melody_string)):
            if num_notes == self.seed_num_notes:
                break
            
            melody_substring.append(melody_string[i])
            if melody_string[i] != "_" and melody_string[i] != "r":
                num_notes += 1

        seed_melody = " ".join(melody_substring)
        print("Input Melody: " + seed_melody)
        return seed_melody

    def temperature_sampling(self, probabilities, temperature):
        """Get index from the probability array, by apllying a softmax function using temperature.
        :param probabilities (nd.array): Probabilities of each possible output.
        :param temperature (float): Value from 0 to 1 that determines the determinism of the model. Higher value, less determinism.
        :return index (int): 
        """

        # + temperature -> smaller values
        predictions = np.log(probabilities) / temperature

        # softmax function 
        # with smaller values the probability distribution is softer -> differences of probabilities will shrink, which leads to a more homogeneous distribution.
        # with larger values the probability distribuition is going to be more conservative -> values with high prob. are more likely to be picked.
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities) # each choice has a probability

        return index

    def generate_melody(self, input_melody, temperature=None):
        """Generate melody with the size of the input_melody.
        :param input_melody (str): Seed encoded melody for the generation.
        :param temperature (float): Value from 0 to 1 that determines the determinism of the model. Higher value, less determinism.
        :return predicted_melody_str (str): Predicted melody
        :return final_melody (str): Input melody + Predicted melody
        """

        pattern = convert_songs_to_int(input_melody)
        predicted_melody = []
        num_steps = len(pattern)

        # generate notes
        step = 0
        for step in range(num_steps):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(get_n_vocab())

            probabilities = self.model.predict(prediction_input, verbose=0)[0]
            if temperature is not None:
                index = self.temperature_sampling(probabilities, temperature)
            else:
                index = np.argmax(probabilities)
            result = self.symbols[index]

            if result == "/":
              break
            
            predicted_melody.append(result)
            pattern.append(index)

        predicted_melody = " ".join(predicted_melody)
        final_melody = input_melody + " " + predicted_melody

        return predicted_melody, final_melody

    def create_midi_file(self, melody, melody_file):
        """Create midi file from the encoded final melody.
        :param final_melody (str): Final encoded melody.
        :param melody_file (str): Melody file path to save the final melody.
        """
        
        midi_stream = m21.stream.Stream()
        
        start_symbol = None
        duration_counter = 1
        melody = melody.split()
        
        # create note and rest objects based on the values generated by the model
        for i, symbol in enumerate(melody):
        
            # symbol is a pitch or rest
            if symbol != '_' or (i+1) == len(melody):
                if start_symbol is not None:
                    if symbol == '_':
                        duration_counter += 1
                    
                    duration = duration_counter * 0.25
                
                    # symbol is a rest
                    if start_symbol == 'r':
                        new_note = m21.note.Rest(quarterLength=duration)
                    # symbol is a note
                    else:
                        new_note = m21.note.Note(int(start_symbol), quarterLength=duration)
                
                    midi_stream.append(new_note)
                    duration_counter = 1
                
                if symbol != '_':
                    start_symbol = symbol
            # symbol is a duration representation
            else:
                duration_counter += 1
        
        if start_symbol is not None:
            duration = duration_counter * 0.25
        
            # symbol is a rest
            if start_symbol == 'r':
                new_note = m21.note.Rest(quarterLength=duration)
            # symbol is a note
            else:
                new_note = m21.note.Note(int(start_symbol), quarterLength=duration)
            
            midi_stream.append(new_note)
        
        midi_stream.write('midi', fp=melody_file)

    def iterative_generation(self):
        """Iteratively generate melodies, using the input melody as the seed and 
        updating the seed of each iteration to the final melody of the previous one.
           In each iteration, a file with its final melody is created.
        """

        input_melody = self.seed_melody
        
        # iterar por n_iterations (n)
        for n in range(self.num_iterations):

            # gerar melodia com base na input_melody (predicted_melody, final_melody)
            predicted_melody, final_melody = self.generate_melody(input_melody, temperature=self.temperature)
            
            if predicted_melody == "":
              break
            
            print("----- Melody generated -----")
            self.create_midi_file(predicted_melody, PREDICTED_MELODY + str(n+1) + ".mid")
            self.create_midi_file(final_melody, FINAL_MELODY + str(n+1) + ".mid")
            print("----- Updated melody file created -----")
            
            # atualizar input_melody para final_melody
            input_melody = final_melody
        
        self.final_melody = final_melody
    
    def iterative_co_generation(self):
        """Iteratively co-generate melodies, using the input melody as the seed and 
        updating the seed of each iteration to the final melody of the previous one.
           In each iteration, a file with its final melody is created, opened with
        Musescore, so the user can change the melody. Then the user has to export and
        substitute the respective MIDI file with the correct one in the Musescore app.
        The seed melody for the next iteration is substituted by the melody in the updated file.
        """
        
        input_melody = self.seed_melody
        
        # iterar por n_iterations (n)
        for n in range(self.num_iterations):
            
            # gerar melodia com base na input_melody (predicted_melody, final_melody)
            predicted_melody, final_melody = self.generate_melody(input_melody, temperature=self.temperature)

            if predicted_melody == "":
              break
            
            print("----- Melody generated -----")
            self.create_midi_file(predicted_melody, PREDICTED_MELODY + str(n+1) + ".mid")
            self.create_midi_file(final_melody, FINAL_MELODY + str(n+1) + ".mid")
            print("----- Predited and final melody files created -----")
            
            melody = m21.converter.parse(FINAL_MELODY + str(n+1) + ".mid")
            melody.show()
            input("----- Enter any key to continue... -----")
            
            input_melody = encode(m21.converter.parse(FINAL_MELODY + str(n+1) + ".mid"))

    def generate(self):
        if self.correction == False:
            self.iterative_generation()
        else:
            self.iterative_co_generation()
      
if __name__ == "__main__":
    
    seed_melody = "ritmos_mais_dificeis.mid"
    seed_num_notes = 17
    seed_pos = "start"
    temperature = 0.5
    num_iterations = 5
    correction = True
    
    mg = MelodyGenerator(seed_melody=seed_melody, seed_num_notes=seed_num_notes, seed_pos=seed_pos, temperature=0.5, num_iterations=num_iterations, correction=correction)
    mg.generate()
    
    ev = Evaluator(target_filename=seed_melody)
    ev.plot_iter_pitch_metrics()
    ev.plot_iter_rythm_metrics()
    ev.plot_final_metrics()
