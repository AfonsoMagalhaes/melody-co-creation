#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:30:08 2021

@author: afonsomagalhaes
"""

import os
import numpy as np
import music21 as m21
import matplotlib.pyplot as plt

from preprocessor import encode

INPUT_DIR = "../input/"
MELODY_DIR = "../melody/"

class Evaluator:
    """Class to calculate and plot the metrics of de co-generated melodies along the iterations."""

    def __init__(self, target_filename=None):
        """Constructor that initializes the Evaluator.
        :param target_filename (str): file name of the melody used to get the seed for the co-generated melody"""
          
          
        self.input_melodies, self.predicted_melodies, self.final_melody = self.get_iter_melodies()
        
        self.metrics_input = {'pc': [], 'pr': [], 'api': [], 'nc': [], 'aioi': []}
        self.metrics_prediction = {'pc': [], 'pr': [], 'api': [], 'nc': [], 'aioi': []}
        self.set_iter_metrics()
        
        if target_filename is not None:
            self.target_melody = self.get_target_melody(target_filename)
        
            if len(self.final_melody) > len(self.target_melody):
                self.final_melody = self.final_melody[:len(self.target_melody)]
            else:
                self.target_melody = self.target_melody[:len(self.final_melody)]
            
                self.metrics_final = {'pc': 0, 'pr': 0, 'api': 0, 'nc': 0, 'aioi': 0}
                self.metrics_target = {'pc': 0, 'pr': 0, 'api': 0, 'nc': 0, 'aioi': 0}
            
                self.set_final_metrics()
        else:
            self.target_melody = None
    
    # calculate all the metrics
    def calc_metrics(self, melody):
        """Calculate the metrics of the melody.
        :param melody (str): Melody to calculate te metrics.
        :return pc (int): Number of different pitches in the melody.
        :return pr (int): Difference between the highest and the lowest pitch in the melody.
        :return api (float): Average value of the difference between two consecutive pitches in the melody.
        :return nc (int): Number of notes in the melody.
        :return aioi (float): Average time interval between the starts of two consecutive notes in the melody.
        """
    
        def pitch_count(melody_pitches):
            """Calculate the number of different pitches in the melody.
            :param melody_pitches (list(int)): List of pitches in the melody.
            :return pc (int):
            """
            
            return len(list(set(melody_pitches)))
        
        # difference between the highest and the lowest pitch
        def pitch_range(melody_pitches):
            """Calculate the difference between the highest and the lowest pitch in the melody.
            :param melody_pitches (list(int)): List of pitches in the melody.
            :return pr (int):
            """
            
            if len(melody_pitches) == 0:
                return 0
            
            return max(melody_pitches) - min(melody_pitches)
        
        # average value between two consecutive pitches
        def average_pitch_interval(melody_pitches):
            """Calculate the average value of the difference between two consecutive pitches in the melody.
            :param melody_pitches (list(int)): List of pitches in the melody.
            :return api (float):
            """
            
            if len(melody_pitches) < 2:
                return 0
            
            pitch_intervals_sum = 0
            for i in range(len(melody_pitches)-1):
                pitch_intervals_sum += abs(melody_pitches[i+1] - melody_pitches[i])
            
            return pitch_intervals_sum / float(len(melody_pitches)-1)
        
        def note_count(melody_pitches):
            """Calculate the number of notes in the melody
            :param melody_pitches (list(int)): List of pitches in the melody.
            :return nc (int):
            """
            
            return len(melody_pitches)
        
        def average_inter_onset_interval(melody, nc):
            """Calculate the average time interval between the starts of two consecutive notes in the melody.
            :param melody (list(str)): List of elements in the melody.
            :param nc (int): number of notes in the melody.
            :return aioi (float):
            """
            
            if nc < 2:
                return 0
            
            melody_copy = melody.copy()
            poped_el = ""
            while (poped_el == "_") or (poped_el == "r"):
                poped_el = melody_copy.pop()
            
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
    
    def get_iter_melodies(self):
        """Get input and predicted melodies from the co-generation iterations and the final result melody.
        :return input_melodies (list(str)): encoded input melodies of all iterations
        :return predicted_melodies (list(str)): encoded predicted melodies of all iterations
        :return final_melody (str): encoded final generated melody"""
        
        input_melodies = []
        predicted_melodies = []
        
        for filename in os.listdir(MELODY_DIR):
            if filename[-3:] == "mid":
                    if filename[:4] == "seed" or filename[:5] == "final":
                        input_melodies.append(encode(m21.converter.parse(MELODY_DIR + filename)))
                    else:
                        predicted_melodies.append(encode(m21.converter.parse(MELODY_DIR + filename)))
        
        final_melody = input_melodies.pop()
        
        return input_melodies, predicted_melodies, final_melody
    
    def get_target_melody(self, target_filename):
        """Get melody used to get the seed for generation.
        :param target_filename (str): filename of the input.
        :return target_mel (str): encoded melody"""
        
        target_mel = encode(m21.converter.parse(INPUT_DIR + target_filename))
        return target_mel
    
    def set_iter_metrics(self):
        """Set metrics of the input and predicted melodies along the co-generation iterations."""
        
        
        for i in range(len(self.input_melodies)):
            pc_input, pr_input, api_input, nc_input, aioi_input = self.calc_metrics(self.input_melodies[i])
            pc_output, pr_output, api_output, nc_output, aioi_output = self.calc_metrics(self.predicted_melodies[i])
            
            # guardar métricas
            self.metrics_input['pc'].append(pc_input)
            self.metrics_input['pr'].append(pr_input)
            self.metrics_input['api'].append(api_input)
            self.metrics_input['nc'].append(nc_input)
            self.metrics_input['aioi'].append(aioi_input)
            
            self.metrics_prediction['pc'].append(pc_output)
            self.metrics_prediction['pr'].append(pr_output)
            self.metrics_prediction['api'].append(api_output)
            self.metrics_prediction['nc'].append(nc_output)
            self.metrics_prediction['aioi'].append(aioi_output)
    
    def set_final_metrics(self):
        """Set metrics of the final co-generated melody and the target melody, used to get the seed."""
        
        pc_final, pr_final, api_final, nc_final, aioi_final = self.calc_metrics(self.final_melody)
        pc_target, pr_target, api_target, nc_target, aioi_target = self.calc_metrics(self.target_melody)
        
        self.metrics_final['pc'] = pc_final
        self.metrics_final['pr'] = pr_final
        self.metrics_final['api'] = api_final
        self.metrics_final['nc'] = nc_final
        self.metrics_final['aioi'] = aioi_final
        
        self.metrics_target['pc'] = pc_target
        self.metrics_target['pr'] = pr_target
        self.metrics_target['api'] = api_target
        self.metrics_target['nc'] = nc_target
        self.metrics_target['aioi'] = aioi_target
    
    
    def plot_iter_pitch_metrics(self):
        """Plot pitch-based metrics of the inout and predicted melodies along the co-generation iterations."""
        
        x = np.arange(1, len(self.input_melodies)+1)
        
        plt.figure(figsize=[20,10])
        plt.title("Pitch-based Metrics")
        plt.plot(x, self.metrics_input['pc'], linestyle='-', marker='o', color='b', label="Pitch Count (Input)")
        plt.plot(x, self.metrics_prediction['pc'], linestyle='--', marker='o', color='b', label="Pitch Count (Output)")
        plt.plot(x, self.metrics_input['pr'], linestyle='-', marker='o', color='r', label="Pitch Range (Input)")
        plt.plot(x, self.metrics_prediction['pr'], linestyle='--', marker='o', color='r', label="Pitch Range (Output)")
        plt.plot(x, self.metrics_input['api'], linestyle='-', marker='o', color='g', label="Average Pitch Interval (Input)")
        plt.plot(x, self.metrics_prediction['api'], linestyle='--', marker='o', color='g', label="Average Pitch Interval (Output)")
        plt.xlabel("Iteration")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()
    
    def plot_iter_rythm_metrics(self):
        """Plot rythm-based metrics of the inout and predicted melodies along the co-generation iterations."""
        
        x = np.arange(1, len(self.input_melodies)+1)
        
        plt.figure(figsize=[20,10])
        plt.title("Rythm-based Metrics")
        plt.plot(x, self.metrics_input['nc'], linestyle='-', marker='o', color='b', label="Note Count (Input)")
        plt.plot(x, self.metrics_prediction['nc'], linestyle='--', marker='o', color='b', label="Note Count (Output)")
        plt.plot(x, self.metrics_input['aioi'], linestyle='-', marker='o', color='g', label="Average Inter-Onset-Interval (Input)")
        plt.plot(x, self.metrics_prediction['aioi'], linestyle='--', marker='o', color='g', label="Average Inter-Onset-Interval (Output)")
        plt.xlabel("Iteration")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()
    
    def plot_final_metrics(self):
        """Plot metrics of the final co-generated melody and the target melody, used to get the seed."""
        
        if self.target_melody is not None:
            pitch_metrics_labels = ["PC", "PR", "API"]
            rythm_metrics_labels = ["NC", "AIOI"]
            
            pitch_metrics_vals_final = [self.metrics_final['pc'], self.metrics_final['pr'], self.metrics_final['api']]
            pitch_metrics_vals_target = [self.metrics_target['pc'], self.metrics_target['pr'], self.metrics_target['api']]
            
            rythm_metrics_vals_final = [self.metrics_final['nc'], self.metrics_final['aioi']]
            rythm_metrics_vals_target = [self.metrics_target['nc'], self.metrics_target['aioi']]
            
            
            plt.figure(figsize=[20,10])
            
            plt.subplot(121)
            plt.title("Pitch-based Metrics")
            plt.bar(pitch_metrics_labels, pitch_metrics_vals_final, label="(Co-)Generated Melody", alpha=0.5)
            plt.bar(pitch_metrics_labels, pitch_metrics_vals_target, label="Target Melody", alpha=0.5)
            plt.xticks(rotation=90)
            plt.ylabel("Metric")
            plt.legend()
            
            plt.subplot(122)
            plt.title("Rythm-based Metrics")
            plt.bar(rythm_metrics_labels, rythm_metrics_vals_final, label="(Co-)Generated Melody", alpha=0.5)
            plt.bar(rythm_metrics_labels, rythm_metrics_vals_target, label="Target Melody", alpha=0.5)
            plt.xticks(rotation=90)
            plt.ylabel("Metric")
            plt.legend()
            
            plt.show()
    
    def plot_target_metrics(self, another_melody):
    
        if self.target_melody is not None:
        
            another_melody = encode(m21.converter.parse(INPUT_DIR + another_melody))
            pc_final, pr_final, api_final, nc_final, aioi_final = self.calc_metrics(another_melody)
            
            plt.figure(figsize=[20,10])
            plt.title("Ordinal Metrics")
            plt.bar(["PC", "PR", "NC"], [self.metrics_target['pc'], self.metrics_target['pr'], self.metrics_target['nc']], label="Do-re-mi", alpha=0.5)
            plt.bar(["PC", "PR", "NC"], [pc_final, pr_final, nc_final], label="Ritmos Difíceis", alpha=0.5)
            plt.xticks(rotation=90)
            plt.ylabel("Metric")
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[20,10])
            plt.title("Average Metrics")
            plt.bar(["API", "AIOI"], [self.metrics_target['api'], self.metrics_target['aioi']], label="Do-re-mi", alpha=0.5)
            plt.bar(["API", "AIOI"], [api_final, aioi_final], label="Ritmos Difíceis", alpha=0.5)
            plt.xticks(rotation=90)
            plt.ylabel("Metric")
            plt.legend()
            plt.show()