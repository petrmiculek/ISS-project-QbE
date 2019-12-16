#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, find_peaks
from scipy.stats import pearsonr

"""
sa1.wav: She had your dark suit in greasy wash water all year.
sa2.wav: Don't ask me to carry an oily rag like that.
si1337.wav: This was chiefly because of the bluish white autofluorescence from the cells.
si1967.wav: Twelve o'clock level.
si707.wav: The avocado should have a give to it, as you hold it, when it is ripe.
sx257.wav: We are open every Monday evening.
sx3.wav: This was easy for us.
sx437.wav: They used an aggressive policeman to flag thoughtless motorists.

sx77.wav: Bagpipes and bongos are musical instruments.                                   <--- q1
sx167.wav: Employee layoffs coincided with the company's reorganization.                 <--- q2

q1.wav: instruments
q2.wav: reorganization
"""


class Record:
    def __init__(self):
        self.name = None

        self.data = None
        self.fs = None
        self.time_axis = None

        self.spectrum_freq = None
        self.spectrum_time = None
        self.spectrum_data = None

        self.spectrum_data_aggr_log = None
        self.spectrum_data_no_dc = None
        self.spectrum_data_aggr = None

        self.scores_q1 = None
        self.scores_q2 = None

        self.peaks1 = None
        self.peaks2 = None


def read_wav_files(path):
    """
    Read in files
    """

    full_path = os.path.join(os.getcwd(), os.path.join(os.pardir, path))

    curr_directory = os.listdir(full_path)

    wav_files = []

    for curr_file in curr_directory:
        if curr_file[-3:] == 'wav':
            file = Record()
            file.data, file.fs = sf.read(os.path.join(full_path, curr_file))
            file.name = curr_file[:-4]

            wav_files.append(file)

    return wav_files


def create_spectrogram(data, fs: int):
    """
        Create spectrogram
        Task 3
    """
    # predefined values given by assignment
    length_per_segment = int(0.025 * fs)  # 400 samples
    length_overlap = int(0.015 * fs)
    nfft = 512

    f, t, sgr = spectrogram(data, fs, nperseg=length_per_segment, noverlap=length_overlap, nfft=nfft)

    return f, t, sgr


def process_file(file):
    """
        Get spectrogram

        Get reduced spectrum matrix precision (compress timeframes)
    """

    file.time_axis = np.linspace(0, len(file.data) / file.fs, num=len(file.data))

    file.spectrum_freq, file.spectrum_time, file.spectrum_data = create_spectrogram(file.data, file.fs)

    # Delete DC signal
    file.spectrum_data_no_dc = np.delete(file.spectrum_data, 0, 0)
    # param hints -->                   (Object, Index, Axis)

    file.spectrum_data_aggr = aggregate_file_data(file)

    file.spectrum_data_aggr_log = 10 * np.log10(file.spectrum_data_aggr + 1e-20)

    return file


def aggregate_file_data(file):
    """
        Reduce spectrum matrix precision
        Task 4
    """
    target_line_count = 16
    divide_total_line_count_by = len(file.spectrum_freq) // target_line_count
    aggregated_data = np.zeros((target_line_count, len(file.spectrum_time)))

    for t in range(0, len(file.spectrum_time)):
        for i in range(0, target_line_count):
            for j in range(0, divide_total_line_count_by):
                aggregated_data[i, t] += file.spectrum_data_no_dc[i * divide_total_line_count_by + j, t]

    return aggregated_data


def do_query(q, s):
    """
        Do one search (query, sentence)
        Task 5
    """
    q_len = len(q.spectrum_data_aggr[0, :])
    s_len = len(s.spectrum_data_aggr[0, :])

    max_index = s_len - q_len - 1

    if max_index < 0:
        print('query shorter than sentence')
        sys.exit(1)

    scores_current = np.zeros(int(np.ceil(s_len / step_size)))
    # last few values (zeroes) are not used, but they make plotting easier

    for j in range(0, max_index, step_size):
        for i in range(0, q_len):
            q_frame = q.spectrum_data_no_dc[:, i]
            s_frame = s.spectrum_data_no_dc[:, i + j]

            tmp, _ = pearsonr(q_frame, s_frame)

            scores_current[j // step_size] += tmp

        scores_current[j // step_size] /= q_len

    return scores_current


def do_queries_by_example(q, all_sentences, query_number):
    """
        Get search results for all sentences (but not for all queries)
        Task 5
    """
    scores = []

    s_count = len(all_sentences)

    for k in range(0, s_count):

        s = all_sentences[k]

        current_score = do_query(q, s)

        scores.append(current_score)
        if query_number == 1:
            s.scores_q1 = current_score
        elif query_number == 2:
            s.scores_q2 = current_score
        else:
            print('invalid query number')
            sys.exit(2)


def find_peaks_in_scores(sentences):
    """
        Evaluate scores of query search
        --> find peaks in scores data
        Task 7
    """
    min_height = 0.5  # alespon polovicni "jistota" = korelace
    pointiness = 0.005  # spise empiricka hodnota
    for file in sentences:
        file.peaks1, properties1 = find_peaks(file.scores_q1, height=min_height, threshold=pointiness)
        file.peaks2, properties2 = find_peaks(file.scores_q2, height=min_height, threshold=pointiness)


def extract_hits_all(sentences, query1, query2):
    """
        Process saving segments of original record
        that correspond to a query hit
        Task 8
    """
    for sentence in sentences:
        extract_hits_from_query(query1, sentence, sentence.peaks1)
        extract_hits_from_query(query2, sentence, sentence.peaks2)


def extract_hits_from_query(query, sentence, peaks):
    """
        Save a segment of original record
        corresponding to a query hit
    """
    for i in range(0, len(peaks)):
        peak = peaks[i]

        hit_start = (peak * len(sentence.time_axis)) * step_size // len(sentence.spectrum_time)
        hit_length = len(query.data)

        sf.write('../' + 'hits/' + '%s_%s_hit%d.wav' % (query.name, sentence.name, i),
                 sentence.data[hit_start:hit_start + hit_length], sentence.fs)


def plot_results(file):
    """
        Plot all results
        Task 6
    """
    fig, (plot_signal, plot_sgr, plot_scores) = plt.subplots(3, 1, figsize=(8, 6))

    """
        Plot signal
    """

    plot_signal.plot(file.time_axis, file.data)
    plot_signal.set_xlabel('t [s]')
    plot_signal.set_ylabel('signal')
    plot_signal.set_ylim([-1, 1])
    plot_signal.set_xlim(left=0, right=max(file.time_axis))
    plot_signal.set_title(file.name)

    """
        Plot signal spectrogram 
    """
    features_x_axis = np.arange(0, 16)

    plot_sgr.pcolormesh(file.spectrum_time, features_x_axis, file.spectrum_data_aggr_log)
    plot_sgr.set_xlabel('t [s]')
    plot_sgr.set_ylabel('features')
    plot_sgr.set_xlim(left=0, right=max(file.time_axis))
    plot_sgr.invert_yaxis()

    """
        Plot query-by-example scores 
    """
    spectrum_time_stepsize = file.spectrum_time[::step_size]

    query1_plot, = plot_scores.plot(spectrum_time_stepsize, file.scores_q1)
    query2_plot, = plot_scores.plot(spectrum_time_stepsize, file.scores_q2)

    plot_scores.set_xlabel('t [s]')
    plot_scores.set_ylabel('scores')
    plot_scores.set_ylim([-0.1, 1])
    plot_scores.set_xlim(left=0, right=max(file.time_axis))

    legend_plots = [query1_plot, query2_plot]
    legend_texts = ['instruments (q1)', 'reorganization (q2)']

    """
        Plot peaks and hits, if any
        peak = beginning of query hit (point -> highest score)
        hit = length of query hit (line -> duration of query)
    """

    if file.peaks1 is not None and len(file.peaks1) > 0:
        peaks1_plot, = plot_scores.plot(spectrum_time_stepsize[file.peaks1], file.scores_q1[file.peaks1], "x")
        q1_len_in_scores = len(query1.spectrum_time) * max(file.time_axis) / (len(file.spectrum_time))

        hits1_plot = plot_scores.hlines(y=file.scores_q1[file.peaks1],
                                        xmin=spectrum_time_stepsize[file.peaks1],
                                        xmax=spectrum_time_stepsize[file.peaks1] + q1_len_in_scores,
                                        colors=peaks1_plot.get_color())

        legend_plots.append(hits1_plot)
        legend_texts.append('q1 hits')

    if file.peaks2 is not None and len(file.peaks2) > 0:
        peaks2_plot, = plot_scores.plot(spectrum_time_stepsize[file.peaks2], file.scores_q2[file.peaks2], "*")
        q2_len_in_scores = len(query2.spectrum_time) * max(file.time_axis) / (len(file.spectrum_time))

        hits2_plot = plot_scores.hlines(y=file.scores_q2[file.peaks2],
                                        xmin=spectrum_time_stepsize[file.peaks2],
                                        xmax=spectrum_time_stepsize[file.peaks2] + q2_len_in_scores,
                                        colors=peaks2_plot.get_color())

        legend_plots.append(hits2_plot)
        legend_texts.append('q2 hits')

    plt.legend(legend_plots,
               legend_texts,
               loc='upper left')

    fig.show()
    fig.savefig('../' + 'plots/' + file.name + '.png')


"""
Main
"""

step_size = 5  # using this one globally

query1, query2 = read_wav_files('queries')
sentences = read_wav_files('sentences')

process_file(query1)
process_file(query2)

for current_file in sentences:
    process_file(current_file)

do_queries_by_example(query1, sentences, 1)
do_queries_by_example(query2, sentences, 2)

find_peaks_in_scores(sentences)

extract_hits_all(sentences, query1, query2)

for sentence in sentences:
    plot_results(sentence)
