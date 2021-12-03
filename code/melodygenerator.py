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
import matplotlib.pyplot as plt
from preprocessor import convert_songs_to_int, transpose, encode, load_mapping
from model import build_model, get_n_vocab

INPUT_MELODY = "../melody/input_melody.mid"
UPDATED_MELODY = "../melody/updated_melody"
FINAL_MELODY = "../melody/final_melody.mid"

class MelodyGenerator:
    
    def __init__(self, input_melody=None, start_melody_length=None, temperature=None, num_iterations=1, correction=False, plots=False):
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.correction = correction
        self.model = build_model(get_n_vocab(), weights="best")
        self.symbols = self.get_vocab_symbols()
        self.final_melody = None
        self.start_melody_length = start_melody_length
        
        if plots == False:
            self.metrics = None
        else:
            self.metrics = {'mel_length': [], 'pc': [], 'pr': [], 'api': [], 'nc': [], 'aioi': []}
        
        if input_melody == None:
            self.input_melody = self.get_input_mel()
        else:
            self.input_melody = self.get_input_mel(input_melody)
            
        

    def get_input_mel(self, filename=None):
        
        # get melody string
        if filename == None:
            music_file = random.choice(os.listdir("../dataset/"))
            with open("../dataset/" + music_file,"r") as fp:
                melody_string = fp.read()
        else:
            with open("../input/" + filename,"r") as fp:
                melody_string = fp.read()
                
        if self.start_melody_length == None or self.start_melody_length > len(melody_string):
            self.start_melody_length = len(melody_string)
            return melody_string

        melody_string = melody_string.split()
        
        # while loop to get correct melody
        melody_found = False
        while melody_found == False:
            
            # get random string of size 12
            random_index = int(random.random() * (len(melody_string) - self.start_melody_length))
            melody_substring = melody_string[random_index:random_index+self.start_melody_length]
            
            # check that string as at least 2 pitches, a pitch at index 0 and no "/"
            if (melody_substring[0] == "_") or (melody_substring[0] == "r"):
                continue

            melody_pitches = list(map(str, filter(lambda symbol: symbol != '_' and symbol != "r", melody_substring)))
            if (len(melody_pitches) < 2) or (not all(melody_pitch in self.symbols for melody_pitch in melody_pitches)):
                continue
            
            melody_found = True
        
        print("Input Melody: " + " ".join(melody_substring))
        return " ".join(melody_substring)
    
    def get_input_melody(self, melody_file):
        return encode(transpose(m21.converter.parse(melody_file)))
    
    def get_updated_melody(self, n):
        return encode(transpose(m21.converter.parse(UPDATED_MELODY + str(n) + ".mid")))

    def get_vocab_symbols(self):
        return list(load_mapping().keys())

    # + temperature -> + unpredictability
    def temperature_sampling(self, probabilities, temperature):
    
        # + temperature -> smaller values
        predictions = np.log(probabilities) / temperature
        
        # softmax function 
        # with smaller values the probability distribution is softer -> differences of probabilities will shrink, which leads to a more homogeneous distribution
        # with larger values the probability distribuition is going to be more conservative -> values with high prob. are more likely to be picked
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
          
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities) # each choice has a probability
          
        return index

    # função de (co-)geração https://www.youtube.com/watch?v=6YdQdf4eBD4&list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz&index=8
    def generate_melody(self, input_melody, temperature=None):
    
        pattern = convert_songs_to_int(self.input_melody)
        predicted_melody = []
        num_steps = len(pattern)

        # generate notes
        for step in range(num_steps):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(get_n_vocab())

            probabilities = self.model.predict(prediction_input, verbose=0)[0]
            if temperature is not None:
                index = self.temperature_sampling(probabilities, temperature)
            else:
                index = np.argmax(probabilities)
            result = self.symbols[index]
    
            if result == '/':
                break
            
            predicted_melody.append(result)
            predicted_melody.append(" ")
            pattern.append(index)

        predicted_melody_str = "".join(predicted_melody)
        final_melody = input_melody + " " + predicted_melody_str
        return predicted_melody_str, final_melody

    def create_midi_file(self, final_melody, melody_file):
        midi_stream = m21.stream.Stream()
        
        start_symbol = None
        duration_counter = 1
        final_melody = final_melody.split()
        
        # create note and rest objects based on the values generated by the model
        for i, symbol in enumerate(final_melody):
        
            # symbol is a pitch or rest
            if symbol != '_' or (i+1) == len(final_melody):
                if start_symbol is not None:
                    duration = duration_counter * 0.25
                
                    # symbol is a rest
                    if start_symbol == 'r':
                        new_note = m21.note.Rest(quarterLength=duration)
                    # symbol is a note
                    else:
                        new_note = m21.note.Note(int(start_symbol), quarterLength=duration)
                
                    midi_stream.append(new_note)
                    duration_counter = 1
                
                start_symbol = symbol
            # symbol is a duration representation
            else:
                duration_counter += 1
        
        midi_stream.write('midi', fp=melody_file)

    # calculate all the metrics
    def calc_metrics(self, melody):
    
        # number of different pitches
        def pitch_count(melody_pitches):
            return len(list(set(melody_pitches)))
        
        # difference between the highest and the lowest pitch
        def pitch_range(melody_pitches):
            if len(melody_pitches) == 0:
                return 0
        
            return max(melody_pitches) - min(melody_pitches)
        
        # average value between two consecutive pitches
        def average_pitch_interval(melody_pitches):
            if len(melody_pitches) < 2:
                return 0
        
            pitch_intervals_sum = 0
            for i in range(len(melody_pitches)-1):
                pitch_intervals_sum += abs(melody_pitches[i+1] - melody_pitches[i])
            
            return pitch_intervals_sum / (len(melody_pitches)-1)
        
        # number of used notes (consider only the rhythm of the notes, not pitches)
        def note_count(melody_pitches):
            return len(melody_pitches)
        
        # calculate the average time between two consecutive notes
        def average_inter_onset_interval(melody, nc):
            if nc < 2:
                return 0
            
            return len(melody) * 0.25 / float(nc - 1)
    
        melody = melody.split()
        melody_pitches = list(map(int, filter(lambda symbol: symbol != '_' and symbol != "r", melody)))
          
        # Pitch based
        pc = pitch_count(melody_pitches)
        pr = pitch_range(melody_pitches)
        api = average_pitch_interval(melody_pitches)
          
        # Rythm based
        nc = note_count(melody_pitches)
        aioi = average_inter_onset_interval(melody, nc)
      
        return pc, pr, api, nc, aioi

    def plot_metrics_generation(self):
        plt.figure(figsize=[20,10])
        plt.title(f"Initial Input Melody Length = {self.metrics['mel_length'][0]}; Num Iterations = {self.num_iterations}")
        plt.plot(self.metrics['mel_length'], self.metrics['pc'], label="Pitch Count")
        plt.plot(self.metrics['mel_length'], self.metrics['pr'], label="Pitch Range")
        plt.plot(self.metrics['mel_length'], self.metrics['api'], label="Average Pitch Interval")
        plt.plot(self.metrics['mel_length'], self.metrics['nc'], label="Note Count")
        plt.plot(self.metrics['mel_length'], self.metrics['aioi'], label="Average Inter-Onset-Interval")
        plt.xlabel("Input Melody Length")
        plt.ylabel("Abs(Pred. Metric - Input Metric)")
        plt.legend()
        plt.show()

    def iterative_generation(self):

        input_melody = self.input_melody
        
        # iterar por n_iterations (n)
        for n in range(self.num_iterations):
        
            # gerar melodia com base na input_melody (predicted_melody, final_melody)
            predicted_melody, final_melody = self.generate_melody(input_melody, temperature=self.temperature)
            
            if predicted_melody == "":
                break
            
            print("Melody generated")
            if self.metrics != None:
                # calcular métricas para input_melody e predicted_melody
                pc_input, pr_input, api_input, nc_input, aioi_input = self.calc_metrics(input_melody)
                pc_output, pr_output, api_output, nc_output, aioi_output = self.calc_metrics(predicted_melody)
            
                # guardar métricas
                self.metrics['mel_length'].append(len(input_melody.split()))
                self.metrics['pc'].append(abs(pc_output - pc_input))
                self.metrics['pr'].append(abs(pr_output - pr_input))
                self.metrics['api'].append(abs(api_output - api_input))
                self.metrics['nc'].append(abs(nc_output - nc_input))
                self.metrics['aioi'].append(abs(aioi_output - aioi_input))
            
            self.create_midi_file(final_melody, UPDATED_MELODY + str(n+1) + ".mid")
            print("Updated melody file created")
            
            # atualizar input_melody para final_melody
            input_melody = final_melody
        
        self.final_melody = final_melody
    
    def iterative_co_generation(self):
        
        input_melody = self.input_melody
        
        # iterar por n_iterations (n)
        for n in range(self.num_iterations):
            
            # gerar melodia com base na input_melody (predicted_melody, final_melody)
            predicted_melody, final_melody = self.generate_melody(input_melody, temperature=self.temperature)
            
            if predicted_melody == "":
                break
            
            print("Melody generated")
            if self.metrics != None:
                # calcular métricas para input_melody e predicted_melody
                pc_input, pr_input, api_input, nc_input, aioi_input = self.calc_metrics(input_melody)
                pc_output, pr_output, api_output, nc_output, aioi_output = self.calc_metrics(predicted_melody)
            
                # guardar métricas
                self.metrics['mel_length'].append(len(input_melody.split()))
                self.metrics['pc'].append(abs(pc_output - pc_input))
                self.metrics['pr'].append(abs(pr_output - pr_input))
                self.metrics['api'].append(abs(api_output - api_input))
                self.metrics['nc'].append(abs(nc_output - nc_input))
                self.metrics['aioi'].append(abs(aioi_output - aioi_input))
            
            self.create_midi_file(final_melody, UPDATED_MELODY + str(n+1) + ".mid")
            print("Updated melody file created")
            
            melody = m21.converter.parse(UPDATED_MELODY + str(n+1) + ".mid")
            melody.show()
            input("Enter any key to continue...")
            
            input_melody = self.get_updated_melody(n+1)
        
        self.final_melody = final_melody
    
    def generate(self):
        if self.correction == False:
            self.iterative_generation()
        else:
            self.iterative_co_generation()
        
        self.plot_metrics_generation()
        
      
if __name__ == "__main__":
    
    # sem melodia de input; 10 iterações; sem correções (geração apenas)
    #mg = MelodyGenerator(start_melody_length=12, num_iterations=10, plots=True)
    
    # com melodia de input; 10 iterações; sem correções
    #mg = MelodyGenerator(input_melody=INPUT_MELODY, start_melody_length=12, num_iterations=10, plots=True)
    
    # sem melodia de input; 10 iterações; com correções
    mg = MelodyGenerator(start_melody_length=12, num_iterations=10, correction=True, plots=True)
    
    # com melodia de input; 10 iterações; com correções
    #mg = MelodyGenerator(input_melody=INPUT_MELODY, start_melody_length=12, num_iterations=10, correction=True, plots=True)
    
    mg.generate()