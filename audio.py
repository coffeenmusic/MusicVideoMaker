import wave
import pyaudio
import numpy as np
from tqdm import tqdm
import os
import pickle

def open_stream(audio_file, CHUNK_MUL=1):
    CHUNK = 1024 * CHUNK_MUL

    wf = wave.open(audio_file, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=RATE,
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

def get_split_times(data, rate, amp_thresh, reset_delta=125, chunk=1024):
    '''
    reset_delta [ms]: length of time (in ms) to wait before a new split can occur
    '''

    reset_delta_frames = int(reset_delta / ((chunk / rate) * 1000)) + 2

    abv_thresh = [np.max(d) > amp_thresh and is_increasing(d) for d in data]
    times = []

    i = 0
    while True:
        if abv_thresh[i] == True:
            times += [i * chunk / rate]
            i += reset_delta_frames
        else:
            i += 1

        if i >= len(abv_thresh):
            # Add final time
            times += [len(data) * chunk / rate]
            break

    return times

def moving_average(x, width=10):
    return np.convolve(x, np.ones(width), 'valid') / width

def is_increasing(data):
    data = moving_average(data, width=400)
    return np.mean(np.diff(data, n=2)) > 0