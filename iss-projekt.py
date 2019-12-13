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
    odkud_vzorky = int(time_start * fs)
    pokud_vzorky = int((time_start + time_how_long) * fs)

    data_segment = data[odkud_vzorky:pokud_vzorky]
    segment_len = data_segment.size

    maxval = (len(data) + fs) / fs
    # print(maxval)

    t = np.linspace(0, maxval, num=len(data))
    # print(t.size)
    plt.figure(figsize=(6, 3))
    plt.plot(t, data)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukovy signal: ')
    plt.tight_layout()
    plt.show()

    return data_segment


def plot_signal_and_spectrum(data, fs, jak_dlouho, odkud):
    """
    Plot segment of signal and its frequency characteristics (spectrum)
    """

    data_segment = plot_signal_segment(data, fs, odkud, jak_dlouho)
    s_seg_spec = np.fft.fft(data_segment)

    G = 10 * np.log10(1 / data_segment.size * np.abs(s_seg_spec) ** 2 + 1e-20)

    t = np.linspace(0, data_segment.size / fs, num=data_segment.size)
    f = np.linspace(0, G.size / data_segment.size * fs, num=G.size)

    _, plots = plt.subplots(2, 1)

    plots[0].plot(t + odkud, data_segment)
    plots[0].set_xlabel('$t[s]$')
    plots[0].set_title('Segment signalu $s$')
    plots[0].grid(alpha=0.5, linestyle='--')

    # zobrazujeme prvni pulku spektra
    plots[1].plot(f[:f.size // 2 + 1], G[:G.size // 2 + 1])
    plots[1].set_xlabel('$f[Hz]$')
    plots[1].set_title('Spektralni hustota vykonu [dB]')
    plots[1].grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()


def create_spectrogram(data, fs):
    length_per_segment = int(0.025 * fs)  # 400 samples
    length_overlap = int(0.015 * fs)
    nfft = 512
    f, t, sgr = spectrogram(data, fs, nperseg=length_per_segment, noverlap=length_overlap, nfft=nfft)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)

    return f, t, sgr


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


def process_file(currentFile):
    res = Record()
    data, fs = sf.read(currentFile)

    # plot_signal_and_spectrum(data, fs, jak_dlouho, odkud)

    sgr_freq, sgr_time, sgr_data = create_spectrogram(data, fs)
    res.spectrum = create_spectrogram_plot(sgr_freq, sgr_time, sgr_data)

    # sgr_freq -> Axis Y
    # sgr_time -> Axis X

    # Delete DC signal

    sgr_data = np.delete(sgr_data, 0, 0)
    #                   (Object, Index, Axis)

    target_line_count = 16

    divide_total_line_count_by = len(sgr_freq) // target_line_count

    data_aggregated_to_line_count = np.zeros((target_line_count, len(sgr_time)))

    for t in range(0, len(sgr_time)):
        for i in range(0, target_line_count):
            for j in range(0, divide_total_line_count_by):
                data_aggregated_to_line_count[i, t] += sgr_data[i*divide_total_line_count_by + j, t]

    # currentFile, data_aggregated_to_line_count, sgr_time
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
    q_len = len(q.data[1, :])
    s_len = len(s.data[1, :])

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

for m in range(0, len(scores1)):
    plot_score(query1.name, sentences[m].name, sentences[m].time_axis, scores1[m])












































