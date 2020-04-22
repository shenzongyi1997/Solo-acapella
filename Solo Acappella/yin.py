# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from audio_processing import *
import time
import pandas as pd


def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x.
    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv



def cumulativeMeanNormalizedDifferenceFunction(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float)
    return np.insert(cmndf, 0, 1)



def getPitch(cmdf, tau_min, tau_max, harmo_th=0.1):
    """
    Return fundamental period of a frame based on CMND function.

    :param cmdf: Cumulative Mean Normalized Difference function
    :param tau_min: minimum period for speech
    :param tau_max: maximum period for speech
    :param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency（Absolute threshold)
    :return: fundamental period if there is values under threshold, 0 otherwise
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0    # if unvoiced



def compute_yin(sig, sr, dataFileName=None, w_len=512, w_step=256, f0_min=100, f0_max=500, harmo_thresh=0.1):
    """

    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.

    :param sig: Audio signal (list of float)
    :param sr: sampling rate (int)
    :param w_len: size of the analysis window (samples)
    :param w_step: size of the lag between two consecutives windows (samples)
    :param f0_min: Minimum fundamental frequency that can be detected (hertz)
    :param f0_max: Maximum fundamental frequency that can be detected (hertz)
    :param harmo_tresh: Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this treshold.

    :returns:

        * pitches: list of fundamental frequencies,
        * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
        * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
        * times: list of time of each estimation
    :rtype: tuple
    """

    print('Yin: compute yin algorithm')
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    timeScale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    times = [t/float(sr) for t in timeScale]
    frames = [sig[t:t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)
    argmins = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):

        #Compute YIN
        df = differenceFunction(frame, w_len, tau_max)
        print(df.shape)
        # plt.figure()
        # plt.plot(df)
        # plt.title("df")
        # plt.show()
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        # plt.figure()
        # plt.plot(cmdf)
        # plt.title("cmdf")
        # plt.show()
        p = getPitch(cmdf, tau_min, tau_max, harmo_thresh)
        # plt.figure()
        # plt.plot(p)
        # plt.title("p")
        # plt.show()
        #Get results
        if np.argmin(cmdf)>tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0: # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else: # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)


    if dataFileName is not None:
        np.savez(dataFileName, times=times, sr=sr, w_len=w_len, w_step=w_step, f0_min=f0_min, f0_max=f0_max, harmo_thresh=harmo_thresh, pitches=pitches, harmonic_rates=harmonic_rates, argmins=argmins)
        print('\t- Data file written in: ' + dataFileName)

    return pitches, harmonic_rates, argmins, times



def appro_note(pitches, duration, t_thresh = 0.1, f_thresh = 5, file_name="./notes.csv"):
    """
        Approximate the notes.

        :param pitches: F0 of each frame (list of float)
        :param duration: duration of each frame (float)
        :param t_thresh: the minimum lasting time for a note(float)
        :param f_thresh: Minimum fundamental frequency change in a note(hertz)
        :param file_name: the csv file that contains realtionship between note and frequency

        :returns:

            * pitch: list of notes
        """
    # if lasting time less than 0.1s, than fit to another
    fs = len(pitches) // duration
    notes = pd.read_csv(file_name)['fre'].tolist()
    pitch = pitches.copy()
    W = int(fs * t_thresh)
    print("pitch length ", len(pitches))
    print("Window size is ",W)
    # overlap = 0.5
    # for i in range((len(pitch) - (W//2)) // (W // 2)):
    #     frame = pitch[i * (W // 2): i * (W//2) + W]
    #     if max(frame) - min(frame) < f_thresh:
    # no overlap smooth
    for i in range(len(pitch)//W):
        frame = pitch[i * W: i * W + W]
        print("frame max ",max(frame)," frame min ", min(frame))
        if max(frame) - min(frame) < f_thresh:
            mean = sum(frame) / W
            for j in range(i*W, i * W + W):
                pitch[j] = mean
        else:
            for j in range(i * W, i * W + W):
                pitch[j] = 0
            # print("set 0!")
    # round to a note
    for i in range(len(pitch) // W):
        frame = pitch[i * W: i * W + W]
        if max(frame) != 0:
            absolute_difference_function = lambda list_value: abs(list_value - max(frame))
            note_fre = min(notes, key=absolute_difference_function)
            for j in range(i * W, i * W + W):
                pitch[j] = note_fre
    return pitch


def output_notes(duration, notes, filename="notes.txt"):
    """
            Output the notes to a txt file.

            :param notes: list of notes
            :param duration: duration of each frame (float)

            :returns:

            """
    file = open(filename,'w')
    cytime = duration / len(notes)
    i = 0
    while i < len(notes):
        if notes[i] == 0:
            i += 1
            continue
        on_time = cytime * i
        fre = notes[i]
        while i < len(notes) - 1 and notes[i + 1] == notes[i] :
            i += 1
        off_time = cytime * ( i + 1)
        file.write(str(on_time) + "\t" + str(fre) + "\t" + str(off_time) + "\n")
        i += 1
    file.close()


def main(audioFileName="scale.wav", w_len=1024, w_step=256, f0_min=80, f0_max=1100, harmo_thresh=0.15, audioDir="./", dataFileName=None, verbose=4):
    """
    Run the computation of the Yin algorithm on a example file.

    Write the results (pitches, harmonic rates, parameters ) in a numpy file.

    :param audioFileName: name of the audio file
    :type audioFileName: str
    :param w_len: length of the window
    :type wLen: int
    :param wStep: length of the "hop" size
    :type wStep: int
    :param f0_min: minimum f0 in Hertz
    :type f0_min: float
    :param f0_max: maximum f0 in Hertz
    :type f0_max: float
    :param harmo_thresh: harmonic threshold
    :type harmo_thresh: float
    :param audioDir: path of the directory containing the audio file
    :type audioDir: str
    :param dataFileName: file name to output results
    :type dataFileName: str
    :param verbose: Outputs on the console : 0-> nothing, 1-> warning, 2 -> info, 3-> debug(all info), 4 -> plot + all info
    :type verbose: int
    """

    if audioDir is not None:
        audioFilePath = audioDir + sep + audioFileName
    else:
        audioFilePath = audioFileName

    sr, sig = audio_read(audioFilePath, formatsox=False)
    duration = len(sig)/float(sr)

    start = time.time()
    pitches, harmonic_rates, argmins, times = compute_yin(sig, sr, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)
    end = time.time()
    notes = appro_note(pitches=pitches, duration=duration, t_thresh=0.1, f_thresh=5, file_name="./notes.csv")
    print("Yin computed in: ", end - start)
    output_notes(duration=duration, notes = notes, filename= "notes.txt")
    mode = 0
    print(type(pitches))
    if mode == 0:
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data')
        ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim([f0_min,f0_max])
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], notes)
        ax3.set_title('notes')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_ylim([f0_min,f0_max])
        plt.savefig("visualization.jpg")
        plt.show()
    if verbose >3 and mode != 0:
        # plt.close()
        plt.figure()
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data')
        ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0')
        ax2.set_ylabel('Frequency (Hz)')
        ax3 = plt.subplot(4, 1, 3, sharex=ax2)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r')
        ax3.set_title('Harmonic rate')
        ax3.set_ylabel('Rate')
        ax4 = plt.subplot(4, 1, 4, sharex=ax2)
        ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins)
        ax4.set_title('Index of minimums of CMND')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (seconds)')
        plt.show()

def compute(audioFileName="scale.wav", w_len=1024, w_step=256, f0_min=80, f0_max=1100, harmo_thresh=0.15, t_thresh = 0.1,audioDir="./", dataFileName=None, verbose=4):
    """
    Run the computation of the Yin algorithm on a example file.

    Write the results (pitches, harmonic rates, parameters ) in a numpy file.

    :param audioFileName: name of the audio file
    :type audioFileName: str
    :param w_len: length of the window
    :type wLen: int
    :param wStep: length of the "hop" size
    :type wStep: int
    :param f0_min: minimum f0 in Hertz
    :type f0_min: float
    :param f0_max: maximum f0 in Hertz
    :type f0_max: float
    :param harmo_thresh: harmonic threshold
    :type harmo_thresh: float
    :param audioDir: path of the directory containing the audio file
    :type audioDir: str
    :param dataFileName: file name to output results
    :type dataFileName: str
    :param verbose: Outputs on the console : 0-> nothing, 1-> warning, 2 -> info, 3-> debug(all info), 4 -> plot + all info
    :type verbose: int
    """

    if audioDir is not None:
        audioFilePath = audioDir + sep + audioFileName
    else:
        audioFilePath = audioFileName
    audioFilePath = audioFileName
    sr, sig = audio_read(audioFilePath, formatsox=False)
    duration = len(sig)/float(sr)

    start = time.time()
    pitches, harmonic_rates, argmins, times = compute_yin(sig, sr, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)
    end = time.time()
    notes = appro_note(pitches=pitches, duration=duration, t_thresh=t_thresh, f_thresh=5, file_name="./notes.csv")
    print("Yin computed in: ", end - start)
    output_notes(duration=duration, notes = notes, filename= "notes.txt")
    mode = 0
    print(type(pitches))
    if mode == 0:
        # plt.close()
        plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data')
        ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim([f0_min,f0_max])
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], notes)
        ax3.set_title('notes')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_ylim([f0_min,f0_max])
        plt.savefig("visualization.jpg")
    if verbose >3 and mode != 0:
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data')
        ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0')
        ax2.set_ylabel('Frequency (Hz)')
        ax3 = plt.subplot(4, 1, 3, sharex=ax2)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r')
        ax3.set_title('Harmonic rate')
        ax3.set_ylabel('Rate')
        ax4 = plt.subplot(4, 1, 4, sharex=ax2)
        ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins)
        ax4.set_title('Index of minimums of CMND')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (seconds)')
        plt.show()



if __name__ == '__main__':
    main(audioFileName="scale.wav", w_len=128, w_step=54, f0_min=400, f0_max=1100)


