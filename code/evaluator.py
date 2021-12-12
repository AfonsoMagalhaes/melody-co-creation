#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:30:08 2021

@author: afonsomagalhaes
"""

import os
import numpy as np
import pandas as pd
import music21 as m21
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from preprocessor import encode

INPUTS_DIR = "../inputs/"
RESULTS_DIR = "../results/"
IMAGES_DIR = "../images/"

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

class Evaluator:
    """Class to calculate and plot the metrics of the co-generated melodies along the iterations."""
    
    def __init__(self):
        """Constructor that initializes the Evaluator."""

        self.metrics_inputs_s_corr = {}
        self.metrics_predictions_s_corr = {}
        self.metrics_finals_s_corr = {}
        
        self.metrics_inputs_c_corr = {}
        self.metrics_predictions_c_corr = {}
        self.metrics_finals_c_corr = {}
        
        self.metrics_originals = {}
        
    # calculate all the metrics
    def calc_metrics(self, melody):
        """Calculate the objective metrics of the melody.
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
        
        def get_average_pitch_intervals(melody_pitches):
            
            if len(melody_pitches) < 2:
                return 0
            
            pitch_intervals = []
            for i in range(len(melody_pitches)-1):
                pitch_intervals.append(abs(melody_pitches[i+1] - melody_pitches[i]))
            
            return pitch_intervals
        
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
        
        def get_inter_onset_intervals(melody):
            
            melody_copy = melody.copy()
            
            # remove initial duration, that have no note at the start
            while melody_copy[0] == "_" or melody_copy[0] == "r":
                melody_copy.pop(0)
            
            inter_onset_intervals = []
            interval = 0.25
            for symbol in melody[1:]:
                if symbol != "_" and symbol != "r":
                    inter_onset_intervals.append(interval)
                    interval = 0.25
                else:
                    interval += 0.25
            
            return inter_onset_intervals
               
        
        melody = melody.split()
        melody_pitches = list(map(int, filter(lambda symbol: symbol != '_' and symbol != "r", melody)))
        
        # Pitch based
        pc = pitch_count(melody_pitches)
        pr = pitch_range(melody_pitches)
        pis = get_average_pitch_intervals(melody_pitches)
          
        # Rythm based
        nc = note_count(melody_pitches)
        iois = get_inter_onset_intervals(melody)
      
        return pc, pr, pis, nc, iois
    
    def set_originals_metrics(self, filename, name):
        
        original_melody = encode(m21.converter.parse(INPUTS_DIR + filename))
        pc_final, pr_final, pis_final, nc_final, iois_final = self.calc_metrics(original_melody)
        self.metrics_originals[name] = {'pc': pc_final,
                                      'pr': pr_final,
                                      'pi': pis_final,
                                      'nc': nc_final,
                                      'ioi': iois_final}
    
    def get_iter_melodies(self, results_dir):
        """Get input and predicted melodies from the co-generation iterations and the final result melody.
        :return input_melodies (list(str)): encoded input melodies of all iterations
        :return predicted_melodies (list(str)): encoded predicted melodies of all iterations
        :return final_melody (str): encoded final generated melody"""
        
        input_filenames = []
        predicted_filenames = []
        seed_melody = None
        
        for filename in os.listdir(RESULTS_DIR + results_dir):
            if filename[-3:] == "mid":
                    if filename[:4] == "seed":
                        seed_melody = encode(m21.converter.parse(RESULTS_DIR + results_dir + filename))
                    elif filename[:5] == "final":
                        input_filenames.append(filename)
                    else:
                        predicted_filenames.append(filename)
        
        input_filenames.sort()
        predicted_filenames.sort()
        
        input_melodies = []
        predicted_melodies = []
        for (input_filename, predicted_filename) in zip(input_filenames, predicted_filenames):
            input_melodies.append(encode(m21.converter.parse(RESULTS_DIR + results_dir + input_filename)))
            predicted_melodies.append(encode(m21.converter.parse(RESULTS_DIR + results_dir + predicted_filename)))
        
        input_melodies.insert(0, seed_melody)
        final_melody = input_melodies.pop()
        
        return input_melodies, predicted_melodies, final_melody
    
    def set_iter_metrics_s_corr(self, results_dir, results_name):
        
        input_melodies, predicted_melodies, final_melody = self.get_iter_melodies(results_dir)
        self.metrics_inputs_s_corr[results_name] = {'pc': [], 'pr': [], 'pi': [], 'nc': [], 'ioi': []}
        self.metrics_predictions_s_corr[results_name] = {'pc': [], 'pr': [], 'pi': [], 'nc': [], 'ioi': []}
        self.metrics_finals_s_corr[results_name] = {'pc': 0, 'pr': 0, 'pi': [], 'nc': 0, 'ioi': []}
        
        
        for i in range(len(input_melodies)):
            pc_input, pr_input, pis_input, nc_input, iois_input = self.calc_metrics(input_melodies[i])
            pc_predicted, pr_predicted, pis_predicted, nc_predicted, iois_predicted = self.calc_metrics(predicted_melodies[i])
            
            # guardar métricas
            self.metrics_inputs_s_corr[results_name]['pc'].append(pc_input)
            self.metrics_inputs_s_corr[results_name]['pr'].append(pr_input)
            self.metrics_inputs_s_corr[results_name]['pi'].append(pis_input)
            self.metrics_inputs_s_corr[results_name]['nc'].append(nc_input)
            self.metrics_inputs_s_corr[results_name]['ioi'].append(iois_input)
            
            self.metrics_predictions_s_corr[results_name]['pc'].append(pc_predicted)
            self.metrics_predictions_s_corr[results_name]['pr'].append(pr_predicted)
            self.metrics_predictions_s_corr[results_name]['pi'].append(pis_predicted)
            self.metrics_predictions_s_corr[results_name]['nc'].append(nc_predicted)
            self.metrics_predictions_s_corr[results_name]['ioi'].append(iois_predicted)
        
        pc_final, pr_final, pis_final, nc_final, iois_final = self.calc_metrics(final_melody)
        self.metrics_finals_s_corr[results_name]['pc'] = pc_final
        self.metrics_finals_s_corr[results_name]['pr'] = pr_final
        self.metrics_finals_s_corr[results_name]['pi'] = pis_final
        self.metrics_finals_s_corr[results_name]['nc'] = nc_final
        self.metrics_finals_s_corr[results_name]['ioi'] = iois_final
    
    def set_iter_metrics_c_corr(self, results_dir, results_name):
        
        input_melodies, predicted_melodies, final_melody = self.get_iter_melodies(results_dir)
        self.metrics_inputs_c_corr[results_name] = {'pc': [], 'pr': [], 'pi': [], 'nc': [], 'ioi': []}
        self.metrics_predictions_c_corr[results_name] = {'pc': [], 'pr': [], 'pi': [], 'nc': [], 'ioi': []}
        self.metrics_finals_c_corr[results_name] = {'pc': 0, 'pr': 0, 'pi': [], 'nc': 0, 'ioi': []}
        
        for i in range(len(input_melodies)):
            pc_input, pr_input, pis_input, nc_input, iois_input = self.calc_metrics(input_melodies[i])
            pc_predicted, pr_predicted, pis_predicted, nc_predicted, iois_predicted = self.calc_metrics(predicted_melodies[i])
            
            # guardar métricas
            self.metrics_inputs_c_corr[results_name]['pc'].append(pc_input)
            self.metrics_inputs_c_corr[results_name]['pr'].append(pr_input)
            self.metrics_inputs_c_corr[results_name]['pi'].append(pis_input)
            self.metrics_inputs_c_corr[results_name]['nc'].append(nc_input)
            self.metrics_inputs_c_corr[results_name]['ioi'].append(iois_input)
            
            self.metrics_predictions_c_corr[results_name]['pc'].append(pc_predicted)
            self.metrics_predictions_c_corr[results_name]['pr'].append(pr_predicted)
            self.metrics_predictions_c_corr[results_name]['pi'].append(pis_predicted)
            self.metrics_predictions_c_corr[results_name]['nc'].append(nc_predicted)
            self.metrics_predictions_c_corr[results_name]['ioi'].append(iois_predicted)
        
        pc_final, pr_final, pis_final, nc_final, iois_final = self.calc_metrics(final_melody)
        self.metrics_finals_c_corr[results_name]['pc'] = pc_final
        self.metrics_finals_c_corr[results_name]['pr'] = pr_final
        self.metrics_finals_c_corr[results_name]['pi'] = pis_final
        self.metrics_finals_c_corr[results_name]['nc'] = nc_final
        self.metrics_finals_c_corr[results_name]['ioi'] = iois_final
    
    def plot_originals_absolute_metrics(self):
        
        plt.figure(figsize=(3.54,3.54), dpi=300)
        
        ind = np.arange(3)
        width = 0.35
        
        i = 0
        for song_name, metrics in self.metrics_originals.items():
            
            plt.bar(ind+width*i, [metrics['pc'], metrics['pr'], metrics['nc']], label=song_name, width=width)
            i += 1

        plt.xticks(ind + width / 2, labels=["PC", "PR", "NC"], rotation=90)
        plt.ylabel("Metric", fontsize=10)
        plt.legend()
        plt.savefig(IMAGES_DIR + 'originals_absolute_metrics.png', bbox_inches='tight')
        plt.show()
        
    def plot_originals_intra_set_metrics(self):
        
        fig, axs = plt.subplots(ncols=2, figsize=(3.54*2,3.54), dpi=300)
        fig.tight_layout()
        
        song_names = []
        num_songs = len (self.metrics_originals)
        apis = []
        aiois = []

        for song_name, metrics in self.metrics_originals.items():
            apis.append(metrics['pi'])
            aiois.append(metrics['ioi'])
            song_names.append(song_name)
        
        axs[0].boxplot(apis)
        axs[0].set_title("Pitch Intervals")
        plt.sca(axs[0])
        plt.xticks(np.arange(1, num_songs+1), song_names, rotation=90)
        plt.ylabel("PI", fontsize=10)
        
        axs[1].boxplot(aiois)
        axs[1].set_title("Inter-Onset-Intervals")
        plt.sca(axs[1])
        plt.xticks(np.arange(1, num_songs+1), song_names, rotation=90)
        plt.ylabel("IOI", fontsize=10)
        
        plt.savefig(IMAGES_DIR + 'originals_intra_set_metrics.png', bbox_inches='tight')
        plt.show()
        
    def plot_iter_absolute_metric_s_corr(self, metric_name):
        
        plt.figure(figsize=(3.54,3.54), dpi=300)
        
        colors = plt.cm.rainbow(np.linspace(0,1,len(self.metrics_inputs_s_corr)))
        iterator = zip(self.metrics_inputs_s_corr.items(), self.metrics_predictions_s_corr.items(), colors)
        for (song_name_input, metrics_input), (song_name_predict, metrics_predict), color in iterator:
            x = np.arange(1, len(metrics_input[metric_name])+1)
            
            plt.plot(x, metrics_input[metric_name], linestyle='-', marker='o', color=color, label=f"{song_name_input} (Input)")
            plt.plot(x, metrics_predict[metric_name], linestyle='--', marker='o', color=color, label=f"{song_name_predict} (Predicted)")
    
        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel(metric_name, fontsize=10)
        plt.legend(fontsize=5, markerscale=0.5)
        
        plt.savefig(IMAGES_DIR + 'iter_' + metric_name + '_s_corr.png', bbox_inches='tight')
        plt.show()
        
    def plot_iter_absolute_metric_c_corr(self, metric_name):

        plt.figure(figsize=(3.54,3.54), dpi=300)

        colors = plt.cm.rainbow(np.linspace(0,1,len(self.metrics_inputs_c_corr)))
        iterator = zip(self.metrics_inputs_c_corr.items(), self.metrics_predictions_c_corr.items(), colors)
        for (song_name_input, metrics_input), (song_name_predict, metrics_predict), color in iterator:
            x = np.arange(1, len(metrics_input[metric_name])+1)
            
            plt.plot(x, metrics_input[metric_name], linestyle='-', marker='o', color=color, label=f"{song_name_input} (Input)")
            plt.plot(x, metrics_predict[metric_name], linestyle='--', marker='o', color=color, label=f"{song_name_predict} (Predicted)")

        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel(metric_name, fontsize=10)
        plt.legend(fontsize=5, markerscale=0.5)
        
        plt.savefig(IMAGES_DIR + 'iter_' + metric_name + '_c_corr.png', bbox_inches='tight')
        plt.show()
        
    def plot_iter_intra_set_metric_s_corr(self, metric_name):
        
        num_songs = len(self.metrics_inputs_s_corr)
        songs_names = list(self.metrics_inputs_s_corr.keys())

        fig, axs = plt.subplots(ncols=num_songs, figsize=(3.54*num_songs,3.54), dpi=300)
        fig.tight_layout()
        
        # iterar pelas músicas
        for n in range(num_songs):
            axs[n].set_title(songs_names[n])
            
            metrics_input = self.metrics_inputs_s_corr[songs_names[n]][metric_name]
            metrics_predicted = self.metrics_predictions_s_corr[songs_names[n]][metric_name]
            
            num_iterations = len(metrics_input)
            
            # em cada música agrupar dados da métrica num dataframe
            # colunas -> [Num iteração, Tipo da melodia (input/prevista), Valor]
            df = pd.DataFrame(columns=["Iteration", "Melody Type", metric_name])
            for i in range(num_iterations):
                
                # put data in the Dataframe format
                metrics_input_iter = metrics_input[i]
                iteration = [i+1 for _ in range(len(metrics_input_iter))]
                melody_type = ["Input" for _ in range(len(metrics_input_iter))]
                
                to_append_input = np.column_stack((iteration, melody_type, metrics_input_iter))
                df_to_append_input = pd.DataFrame(to_append_input, columns=["Iteration", "Melody Type", metric_name])
                
                metrics_predicted_iter = metrics_predicted[i]
                iteration = [i+1 for _ in range(len(metrics_predicted_iter))]
                melody_type = ["Predicted" for _ in range(len(metrics_predicted_iter))]
                
                to_append_predicted = np.column_stack((iteration, melody_type, metrics_predicted_iter))
                df_to_append_predicted = pd.DataFrame(to_append_predicted, columns=["Iteration", "Melody Type", metric_name])
                
                # append data to final Dataframe
                df = df.append(df_to_append_input)
                df = df.append(df_to_append_predicted)
            
            if metric_name == "pi":
                df = df.astype({'Iteration': 'int64', 'Melody Type': 'object', metric_name: 'int64'})
            elif metric_name == "ioi":
                df = df.astype({'Iteration': 'int64', 'Melody Type': 'object', metric_name: 'float64'})
            
            sns.boxplot(x="Iteration", y=metric_name, hue="Melody Type", data=df, notch=True, ax=axs[n])
            plt.sca(axs[n])
            plt.legend(fontsize=5, markerscale=0.5)
        
        plt.savefig(IMAGES_DIR + 'iter_' + metric_name + '_s_corr.png', bbox_inches='tight')
        plt.show()
          
    def plot_iter_intra_set_metric_c_corr(self, metric_name):
        
        num_songs = len(self.metrics_inputs_c_corr)
        songs_names = list(self.metrics_inputs_c_corr.keys())

        fig, axs = plt.subplots(ncols=num_songs, figsize=(3.54*num_songs,3.54), dpi=300)
        fig.tight_layout()
        
        # iterar pelas músicas
        for n in range(num_songs):
            axs[n].set_title(songs_names[n])
            
            metrics_input = self.metrics_inputs_c_corr[songs_names[n]][metric_name]
            metrics_predicted = self.metrics_predictions_c_corr[songs_names[n]][metric_name]
            
            num_iterations = len(metrics_input)
            
            # em cada música agrupar dados da métrica num dataframe
            # colunas -> [Num iteração, Tipo da melodia (input/prevista), Valor]
            df = pd.DataFrame(columns=["Iteration", "Melody Type", metric_name])
            for i in range(num_iterations):
                
                # put data in the Dataframe format
                metrics_input_iter = metrics_input[i]
                iteration = [i+1 for _ in range(len(metrics_input_iter))]
                melody_type = ["Input" for _ in range(len(metrics_input_iter))]
                
                to_append_input = np.column_stack((iteration, melody_type, metrics_input_iter))
                df_to_append_input = pd.DataFrame(to_append_input, columns=["Iteration", "Melody Type", metric_name])
                
                metrics_predicted_iter = metrics_predicted[i]
                iteration = [i+1 for _ in range(len(metrics_predicted_iter))]
                melody_type = ["Predicted" for _ in range(len(metrics_predicted_iter))]
                
                to_append_predicted = np.column_stack((iteration, melody_type, metrics_predicted_iter))
                df_to_append_predicted = pd.DataFrame(to_append_predicted, columns=["Iteration", "Melody Type", metric_name])
                
                # append data to final Dataframe
                df = df.append(df_to_append_input)
                df = df.append(df_to_append_predicted)
            
            if metric_name == "pi":
                df = df.astype({'Iteration': 'int64', 'Melody Type': 'object', metric_name: 'int64'})
            elif metric_name == "ioi":
                df = df.astype({'Iteration': 'int64', 'Melody Type': 'object', metric_name: 'float64'})
            
            sns.boxplot(x="Iteration", y=metric_name, hue="Melody Type", data=df, notch=True, ax=axs[n])
            plt.sca(axs[n])
            plt.legend(fontsize=5, markerscale=0.5)
        
        plt.savefig(IMAGES_DIR + 'iter_' + metric_name + '_c_corr.png', bbox_inches='tight')
        plt.show()
        
if __name__ == "__main__":
    
    ev = Evaluator()
    
    ev.set_originals_metrics("do_re_mi.mid", "Do-Re-Mi")
    ev.set_originals_metrics("blood_rose.mid", "Blood-Rose")
    
    ev.set_iter_metrics_s_corr("do_re_mi_s_corr/", "Do-Re-Mi")
    ev.set_iter_metrics_s_corr("blood_rose_s_corr/", "Blood-Rose")
    ev.set_iter_metrics_s_corr("blood_rose_s_corr_input_maior/", "Blood-Rose (Seed Maior)")
    
    ev.set_iter_metrics_c_corr("do_re_mi_c_corr/", "Do-Re-Mi")
    ev.set_iter_metrics_c_corr("blood_rose_c_crr/", "Blood-Rose (Seed Maior)")
    
    ev.plot_originals_absolute_metrics()
    ev.plot_originals_intra_set_metrics()
    
    ev.plot_iter_absolute_metric_s_corr("pc")
    ev.plot_iter_absolute_metric_s_corr("pr")
    ev.plot_iter_absolute_metric_s_corr("nc")
    
    ev.plot_iter_intra_set_metric_s_corr("pi")
    ev.plot_iter_intra_set_metric_s_corr("ioi")
    
    ev.plot_iter_absolute_metric_c_corr("pc")
    ev.plot_iter_absolute_metric_c_corr("pr")
    ev.plot_iter_absolute_metric_c_corr("nc")
    
    ev.plot_iter_intra_set_metric_c_corr("pi")
    ev.plot_iter_intra_set_metric_c_corr("ioi")
    