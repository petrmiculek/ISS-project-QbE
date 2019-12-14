import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
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
    # def __init__(self, file_name, data, time_axis, spectrum, signal, scores_q1, scores_q2):
    def __init__(self):
        self.name = None
        self.data = None
        self.time_axis = None
        self.spectrum = None
        self.signal = None
        self.scores_q1 = None
        self.scores_q2 = None


def prep_files():
    curr_directory = os.listdir(os.getcwd())
    curr_directory.sort()

    for curr_file in curr_directory:
        if curr_file[-3:] != 'wav':
            curr_directory.remove(curr_file)

    # TODO remove in submission
    curr_directory.remove('Pipfile')
    curr_directory.remove('.gitignore')

    return curr_directory


def plot_signal_segment(data, fs, time_start, time_how_long):
    """
        Plot segment of signal
    """

    time_end = time_start + time_how_long

    samples_start = int(time_start * fs)
    samples_count = int(time_how_long * fs)

    samples_end = samples_start + samples_count

    if len(data) < samples_end:
        print('Warning: segment out of data bounds')
        exit(1)

    data_segment = data[samples_start:samples_start + samples_count]

    t = np.linspace(time_start, time_end, num=samples_count)

    plt.figure(figsize=(6, 3))
    plt.plot(t, data_segment)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukovy signal[%f:%f]: ' % (time_start, time_end))
    plt.tight_layout()
    plt.show()


"""
def plot_signal_whole(data, fs: int) -> plt:
    ""\"
        Plot whole signal
    ""\"

    t = np.linspace(0, len(data) / fs, num=len(data))

    plt.figure(figsize=(6, 3))
    plt.plot(t, data)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukovy signal:')
    plt.tight_layout()
"""


def plot_signal_spectrum(data, fs: int) -> plt:
    """
    Plot segment of signal and its frequency characteristics (spectrum)
    """

    data_ffted = np.fft.fft(data)

    data_log_scale = 10 * np.log10(1 / data.size * np.abs(data_ffted) ** 2 + 1e-20)

    freq_axis = np.linspace(0, data_log_scale.size / data.size * fs, num=data_log_scale.size)

    # zobrazujeme prvni pulku spektra
    plt.plot(freq_axis[:freq_axis.size // 2 + 1], data_log_scale[:data_log_scale.size // 2 + 1])
    plt.xlabel('$f[Hz]$')
    plt.title('Spektralni hustota vykonu [dB]')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    # plt.show()
    return plt


def create_spectrogram(data, fs: int):
    length_per_segment = int(0.025 * fs)  # 400 samples
    length_overlap = int(0.015 * fs)
    nfft = 512
    f, t, sgr = spectrogram(data, fs, nperseg=length_per_segment, noverlap=length_overlap, nfft=nfft)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)

    return f, t, sgr


"""
def create_spectrogram_plot(f, t, sgr):
    sgr_log = 10 * np.log10(sgr + 1e-20)
    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Cas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralni hustota vykonu [dB]', rotation=270, labelpad=15)
    plt.tight_layout()

    return plt
"""


def process_file(currentFile):
    res = Record()
    data, fs = sf.read(currentFile)

    fig, (plot_signal, plot_sgr, plot_scores) = plt.subplots(3)

    """
    Plot signal
    """
    t = np.linspace(0, len(data) / fs, num=len(data))

    plot_signal.plot(t, data)
    plot_signal.set_xlabel('$t [s]$')
    plot_signal.set_ylim([-1, 1])
    plot_signal.set_title(currentFile)

    """
    Signal spectrogram 
    """
    sgr_freq, sgr_time, sgr_data = create_spectrogram(data, fs)
    sgr_log = 10 * np.log10(sgr_data + 1e-20)

    # sgr_freq -> Axis Y
    # sgr_time -> Axis X

    plot_sgr.pcolormesh(sgr_time, sgr_freq, sgr_log)
    plot_sgr.set_xlabel('$t [s]$')
    plot_sgr.set_ylabel('$f [Hz]$')

    fig.tight_layout()
    fig.show()

    # Delete DC signal

    sgr_data = np.delete(sgr_data, 0, 0)
    #                   (Object, Index, Axis)

    """
    Reduce spectrum matrix precision (compress timeframes)
    """
    target_line_count = 16

    divide_total_line_count_by = len(sgr_freq) // target_line_count

    data_aggregated_to_line_count = np.zeros((target_line_count, len(sgr_time)))

    for t in range(0, len(sgr_time)):
        for i in range(0, target_line_count):
            for j in range(0, divide_total_line_count_by):
                data_aggregated_to_line_count[i, t] += sgr_data[i * divide_total_line_count_by + j, t]

    res.name = current_file
    res.time_axis = sgr_time
    res.data = data_aggregated_to_line_count

    return res


def do_queries_by_example(q, all_sentences, query_number):

    scores = []

    s_count = len(all_sentences)

    for k in range(0, s_count):

        s = all_sentences[k]
        # print(s.name, ':', len(s.data[1, :]))

        current_score = do_query(q, s)

        scores.append(current_score)
        if query_number == 1:
            s.scores_q1 = current_score
        elif query_number == 2:
            s.scores_q1 = current_score
        else:
            print('invalid query number')

    # currently not needed, scores are saved within Record objects
    return scores


def do_query(q, s):
    q_len = len(q.data[0, :])
    s_len = len(s.data[0, :])

    max_index = s_len - q_len - 1

    if max_index < 0:
        print('query shorter than sentence')
        return None

    scores_current = np.zeros(s_len)
    # last q_len values (zeroes) are not used, but they make plotting easier

    for j in range(0, max_index):
        for i in range(0, len(q.data)):

            q_frame = q.data[:, i]
            s_frame = s.data[:, i + j]

            if q_frame.any() and s_frame.any():
                tmp, _ = pearsonr(q_frame, s_frame)
            else:
                continue

            if np.math.isnan(tmp):
                continue

            scores_current[j] += tmp

        scores_current[j] /= q_len

    print(s.name + ' max=:', max(scores_current), 'min:', min(scores_current))
    return scores_current


def plot_score(query_name, sentence_name, sentence_time_axis, score):

    plt.plot(sentence_time_axis, score)

    plt.gca().set_xlabel('Cas [s]')
    plt.gca().set_ylabel('Podobnost [nevimco]')
    plt.ylim([-0.1, 1])
    plt.title('%s in %s' % (query_name, sentence_name))

    # plt.tight_layout()

    # plt.show()
    return plt


"""
Main
"""

"""
TODO
step size in QbE
"""
query1_name = 'q1.wav'
query2_name = 'q2.wav'

sentences = []  # list of Record objects
query1 = Record()
query2 = Record()

curr_dir = prep_files()

for current_file in curr_dir:
    if current_file == query1_name:
        query1 = process_file(current_file)
    elif current_file == query2_name:
        query2 = process_file(current_file)
    else:
        sentences.append(process_file(current_file))

scores1 = do_queries_by_example(query1, sentences, 1)
# scores2 = do_queries_by_example(query2, sentences, 2)

# for i in sentences:
#     show all their plots

for m in range(0, len(scores1)):
    plot_score(query1.name, sentences[m].name, sentences[m].time_axis, scores1[m])
