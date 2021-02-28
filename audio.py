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

# def filter_audio_time(data, rate, chunk, start_time=None, stop_time=None):
#     seconds_per_chunk = chunk / rate
#     start_idx = int(start_time/seconds_per_chunk) if start_time else 0
#     stop_idx = int(stop_time/seconds_per_chunk) if stop_time else len(data) - 1
#
#     # Update times for rounding during conversion to integer index
#     new_start_time = start_idx * seconds_per_chunk
#     new_stop_time = stop_idx * seconds_per_chunk
#
#     return data[start_idx:stop_idx], new_start_time, new_stop_time

def get_split_times(data, rate, amp_thresh, min_reset=125, chunk=1024, start_time=0, stop_time=0):
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