import wave
import pyaudio
from spleeter.separator import Separator
from spleeter.audio import STFTBackend
import numpy as np
from tqdm import tqdm
import os
import sys
import pickle

SEPARATE_DICT = {0: 'drums.wav', 1: 'bass.wav', 2: 'vocals.wav', 3: 'other.wav'}

def open_stream(audio_file, CHUNK_MUL=1):
    CHUNK = 1024 * CHUNK_MUL

    wf = wave.open(audio_file, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    return stream, wf, CHUNK

def get_saved_audio(file):
    audio_pkl_filename = file.split('.')[0] + '.pkl'
    if os.path.exists(audio_pkl_filename):
        print('Saved audio data exists. Skipping preprocessing...')
        saved_data = pickle.load(open(audio_pkl_filename, "rb"))
        audio_data = saved_data['data']
        CHUNK = saved_data['chunk']
        RATE = saved_data['rate']

        return audio_data, CHUNK, RATE

def get_audio_data(file, save=True):
    stream, wf, CHUNK = open_stream(file)
    RATE = wf.getframerate()

    with tqdm(total=wf.getnframes()) as pbar:

        cnt = 0
        while True:
            # Read next frame
            data = wf.readframes(CHUNK)
            pbar.update(CHUNK)
            if len(data) < CHUNK:
                break

            data_int = np.frombuffer(data, dtype=np.int32)  # Read bytes to int
            data_int = np.resize(data_int, (1, CHUNK))  # Handle final CHUNK where size might be less than CHUNK size

            if cnt == 0:
                all_data = data_int.copy()
            else:
                all_data = np.append(all_data, data_int, axis=0)

            cnt += 1

    if save:
        audio_pkl_filename = file.split('.')[0] + '.pkl'
        pickle.dump({'data': all_data, 'chunk': CHUNK, 'rate': RATE}, open(audio_pkl_filename, "wb"))

    return all_data, CHUNK, RATE

def fft_to_buckets(freq, PSD, buckets):
    """
    Takes the current CHUNK's frequency response and breaks each frequency in to buckets
    - freq: audio files CHUNK of data amplitudes converted in to frequencies
    - PSD: power spectral density of each frequency
    - buckets: a list of frequencies where each freq in the list will create a range between that freq and the previous
        example - [100, 1000, 5000] Hz
    """
    idxs = sorted({np.abs(freq - i).argmin() for i in buckets}) # Get indices of freq from closest frequencies in buckets

    # Average PSD values in between frequencies defined by buckets
    freq_bucket = [PSD[idxs[i]:idxs[i+1]].mean() for i in range(len(idxs)-1)]  + [PSD[idxs[-1]:].mean()]

    return freq_bucket, idxs

def get_minmax_bucket_freq(audio_data, buckets, rate):
    for i, data in enumerate(audio_data):
        n = len(data)
        fhat = np.fft.fft(data, n)
        PSD = np.abs(fhat * np.conj(fhat) / n)  # Power Spectral Density
        freq = (rate / n) * np.arange(n)

        fb, idxs = fft_to_buckets(freq, PSD, buckets)  # Chunk frequencies in to buckets

        if i == 0:
            all_buckets = np.array(fb)
        else:
            all_buckets = np.vstack((all_buckets, fb))

    return np.min(all_buckets, axis=0), np.max(all_buckets, axis=0)

def get_audio_freqs_in_buckets(audio_data_chunk, buckets, rate):
    """
        Takes the current CHUNK's audio amplitudes, converts to the frequency domain, then buckets those frequencies
        - buckets: a list of frequencies where each freq in the list will create a range between that freq and the previous
            example - [100, 1000, 5000] Hz
    """

    n = len(audio_data_chunk)
    fhat = np.fft.fft(audio_data_chunk, n)
    PSD = np.abs(fhat * np.conj(fhat) / n)  # Power Spectral Density
    freq = (rate / n) * np.arange(n)

    fb, _ = fft_to_buckets(freq, PSD, buckets)  # Chunk frequencies in to buckets
    return fb

def get_split_times(data, rate, thresholds, buckets, buckets_min, buckets_max, min_reset=125, chunk=1024, start_time=0, stop_time=0):
    '''
    min_reset [ms]: length of time (in ms) to wait before a new split can occur
    start_time[s]: start audio data here
    stop_time [s]: stop audio data here
    '''
    stop_time = len(data) * (chunk / rate) if stop_time == 0 else stop_time

    min_reset_frame_cnt = int(min_reset / ((chunk / rate) * 1000)) + 2

    times = [start_time]

    i = 0
    while True:
        time = i * chunk / rate

        freq_buckets = get_audio_freqs_in_buckets(data[i], buckets, rate)

        # Scale buckets
        scaled = (freq_buckets - buckets_min) / (buckets_max - buckets_min)

        # If freq range is above threshold limit. Ignore thresholds set to 0
        abv_thresh = any([(s > thresholds[s_idx]) and thresholds[s_idx] > 0 for s_idx, s in enumerate(scaled)])

        # Filter to start & stop times
        if time >= start_time and time <= stop_time:
            if abv_thresh:
                times += [time]
                i += min_reset_frame_cnt
            else:
                i += 1
        else:
            i += 1

        # If No More data
        if i >= len(data):
            # Add final time
            times += [stop_time]
            break

    return times

def get_split_times_simple(data, rate, amp_thresh, min_reset=125, chunk=1024, start_time=0, stop_time=0):
    '''
    min_reset [ms]: length of time (in ms) to wait before a new split can occur
    start_time[s]: start audio data here
    stop_time [s]: stop audio data here
    '''
    stop_time = len(data) * (chunk / rate) if stop_time == 0 else stop_time
    # if start_time > 0 or stop_time > 0:
    #     data, _, _ = filter_audio_time(data, rate, chunk, start_time, stop_time)

    min_reset_frame_cnt = int(min_reset / ((chunk / rate) * 1000)) + 2

    abv_thresh = [np.max(d) > amp_thresh and is_increasing(d) for d in data]
    times = [start_time]

    i = 0
    while True:
        time = i * chunk / rate

        # Filter to start & stop times
        if time >= start_time and time <= stop_time:
            if abv_thresh[i] == True:
                times += [time]
                i += min_reset_frame_cnt
            else:
                i += 1
        else:
            i += 1

        if i >= len(abv_thresh):
            # Add final time
            times += [stop_time]
            break

    return times

def moving_average(x, width=10):
    return np.convolve(x, np.ones(width), 'valid') / width

def is_increasing(data):
    data = moving_average(data, width=400)
    return np.mean(np.diff(data, n=2)) > 0

def separate_audio_tracks(audio_file, save_dir=None, use_gpu=True):
    if not save_dir:
        save_dir = "Media\\Audio\\Separated"

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    audio_filename = audio_file.split('\\')[-1]
    separated_path = os.path.join(save_dir, audio_filename.split('.')[0])
    if os.path.exists(separated_path) and len(os.listdir(separated_path)) > 0:
        print('Separated tracks found. Skipping audio track separation.')
    else:
        print('Separating music in to drums, bass, vocals, & other and saving in save_dir.')
        mult = False if 'win' in sys.platform else True # Handle windows lack of support for multiprocess module
        separator = Separator('spleeter:4stems', multiprocess=mult)  # Split to: Bass, Drums, Vocals, & Other
        separator.separate_to_file(audio_file, save_dir, synchronous=True)

    return os.path.join(save_dir, audio_filename.split('.')[0])