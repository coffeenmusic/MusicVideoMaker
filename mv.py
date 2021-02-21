from moviepy.editor import *
from decord import VideoReader
from decord import cpu, gpu
#import cv2
import wave
import pyaudio
from tqdm import tqdm
import numpy as np
import sys
import os
import time
import pickle

"""
TODO:
- Continuous Mode with shuffle option
- Save audio data for reuse
"""

VID_DIR = os.path.join('Media', 'Videos') # Default video directory
EXPORT_FILENAME = 'music_video.mp4'
SHUFFLE_CNT = 0
SHUFFLE_CHUNK_SIZE = 20
CHECK_FREQ = 1
USE_DECORD = False
amp_thresh = 1000000000
#amp_thresh = 150000000
args = sys.argv

# Print Help
if len(args) == 1:
    print('Auto Music Video Maker Commands:')
    print('-a Audio\\Path\\ref_audio.wav')
    print('-m Audio\\Path\\final_music.wav')
    print('-v Video\\Set\\Dir\\')
    print('-n export_filename.mp4')
    exit(0)

# 0 - python directory/script.py
# 1 - video directory
# 2 - audio reference
# 3 - final music
i = 0
while True:
    if args[i] == '-v':
        i += 1
        VID_DIR = str(args[i])
    elif args[i] == '-a':
        i += 1
        AUD_FILE = str(args[i])
        FINAL_AUDIO = AUD_FILE
    elif args[i] == '-m':
        i += 1
        FINAL_AUDIO = str(args[i])
    elif args[i] == '-n':
        i += 1
        EXPORT_FILENAME = str(args[i])
    elif args[i] == '-t':
        i += 1
        amp_thresh = int(args[i])
    elif args[i] == '-hs':
        USE_DECORD = True
    elif args[i] == '-shuffle':
        i += 1
        SHUFFLE_CNT = int(args[i])
    elif args[i] == '-freq':
        i += 1
        CHECK_FREQ = float(args[i])

    i += 1
    if i >= len(args):
        break

VID_FILES = [os.path.join(VID_DIR, f) for f in os.listdir(VID_DIR) if f.split('.')[-1].lower() in ['mp4', 'avi', 'mkv', 'm4v']]

print('Video Directory: ', VID_DIR)
print('Reference Audio File: ', AUD_FILE)
print('Song Used in Final Music Video: ', FINAL_AUDIO)
print('Export Filename: ', EXPORT_FILENAME)

start_time = time.time()

"""
CUT VIDEO IN TO CLIP FUNCTIONS
"""

def scene_changed(prev_frame, frame, delta_thresh=30):
    delta = abs(np.mean(prev_frame) - np.mean(frame))

    if delta > delta_thresh:
        return True
    return False


def split_video_decord(vid_filename, check_freq=1, print_split_frames=False, print_cmp_frames=False):
    """
    print_split_frames - for troubleshooting, may remove later
    print_cmp_frames - for troubleshooting, may remove later
    max_clips - set above 0 to stop early when len(clips) greater than max_clips
    check_freq [seconds] - how often to compare two frames for scene change
    """
    with open(vid_filename, 'rb') as f:
        video = VideoFileClip(vid_filename)
        vr = VideoReader(vid_filename, ctx=cpu(0))

        clip_cnt = 0  # Number of clips created from video
        start_time = 0  # time in seconds from video where current clip starts
        clips = []  # list of subclips of video file created by video split

        frame_freq = int(video.reader.fps * check_freq)
        print(f'Compare frames every {check_freq} seconds. This equals {frame_freq} frames.')

        total_frames = len(vr)

        prev_frame = vr[0]

        no_change_cnt = 0
        for i in range(0, total_frames, frame_freq):
            frame = vr[i].asnumpy()

            if print_cmp_frames:
                print_frame(np.append(prev_frame, frame, axis=1))

            if i > 0:  # Skip first frame
                if start_time != stop_time and scene_changed(prev_frame, frame, delta_thresh=20):
                    if print_split_frames:
                        print_frame(prev_frame)

                    clip = video.subclip(start_time, stop_time)

                    start_time = i / video.reader.fps

                    clip_cnt += 1

                    yield clip
                else:
                    no_change_cnt += 1

                if no_change_cnt > 10:
                    cont = input('Not seeing scene changes. Continue [y/n]?')
                    if cont.lower() not in ['y', 'yes']:
                        break
            prev_frame = frame
            stop_time = i / video.reader.fps

def split_video(vid_filename, check_freq=1, print_split_frames=False, print_cmp_frames=False):
    """
    print_split_frames - for troubleshooting, may remove later
    print_cmp_frames - for troubleshooting, may remove later
    max_clips - set above 0 to stop early when len(clips) greater than max_clips
    check_freq [seconds] - how often to compare two frames for scene change
    """
    clip_cnt = 0  # Number of clips created from video
    start_time = 0  # time in seconds from video where current clip starts
    clips = []  # list of subclips of video file created by video split
    video = VideoFileClip(vid_filename)

    frame_freq = int(video.reader.fps * check_freq)
    print(f'Compare frames every {check_freq} seconds. This equals {frame_freq} frames.')

    prev_frame = video.get_frame(0)  # Initialize previous frame

    for i, (time, frame) in tqdm(enumerate(video.iter_frames(with_times=True))):

        if i % frame_freq == 0:
            if print_cmp_frames:
                print_frame(np.append(prev_frame, frame, axis=1))

            if i > 0:  # Skip first frame
                if start_time != stop_time and scene_changed(prev_frame, frame, delta_thresh=20):
                    if print_split_frames:
                        print_frame(prev_frame)

                    yield video.subclip(start_time, stop_time)

                    start_time = time

                    clip_cnt += 1

            prev_frame = frame.copy()
            stop_time = time
    return clips

"""
AUDIO FUNCTIONS
"""

def open_stream(audio_file, CHUNK_MUL=1):
    CHUNK = 1024 * CHUNK_MUL

    wf = wave.open(audio_file, 'rb')
    RATE = wf.getframerate()
    FPS = RATE / CHUNK

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=RATE,
                    output=True)

    return stream, wf, CHUNK, RATE


def get_audio_data():
    stream, wf, CHUNK, RATE = open_stream(AUD_FILE)

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

    return all_data, CHUNK, RATE


audio_pkl_filename = AUD_FILE.split('.')[0] + '.pkl'
if os.path.exists(audio_pkl_filename):
    print('Saved audio data exists. Skipping preprocessing...')
    saved_data = pickle.load(open(audio_pkl_filename, "rb" ))
    audio_data = saved_data['data']
    CHUNK = saved_data['chunk']
    RATE = saved_data['rate']
else:
    print('Preprocessing Audio...')
    audio_data, CHUNK, RATE = get_audio_data()
    pickle.dump({'data': audio_data, 'chunk': CHUNK, 'rate': RATE}, open(audio_pkl_filename, "wb"))

def moving_average(x, width=10):
    return np.convolve(x, np.ones(width), 'valid') / width

def is_increasing(data):
    data = moving_average(data, width=400)
    return np.mean(np.diff(data, n=2)) > 0

def get_split_times(data, reset_delta=125, chunk=CHUNK, rate=RATE):
    '''
    reset_delta [ms]: length of time (in ms) to wait before a new split can occur
    '''

    reset_delta_frames = int(reset_delta / ((chunk / rate) * 1000)) + 2

    abv_thresh = [np.max(d) > amp_thresh and is_increasing(d) for d in audio_data]
    times = []

    i = 0
    while True:
        if abv_thresh[i] == True:
            times += [i * CHUNK / RATE]
            i += reset_delta_frames
        else:
            i += 1

        if i >= len(abv_thresh):
            # Add final time
            times += [len(audio_data) * CHUNK / RATE]
            break

    return times

"""
BUILD MUSIC VIDEO FUNCTIONS
"""
# def build_mv_clips(times):
#     with tqdm(total=len(times)) as pbar:
#         cut_lens = np.diff([0] + times)
#
#         clips = []
#         mv_clips = []
#
#         split_func = split_video_decord if USE_DECORD else split_video
#
#         cut_len = cut_lens[0]
#         for video in VID_FILES:
#             print(video)
#             for clip in split_func(video):
#                 clip_len = clip.duration
#                 if clip_len > cut_len:
#                     mv_clips += [clip.subclip(0, cut_len)]
#                     if len(mv_clips) < len(cut_lens):
#                         cut_len = cut_lens[len(mv_clips)]
#                         pbar.update(1)
#                     else:
#                         return mv_clips
#         return mv_clips

def build_mv_clips(times, clip_generator):

    with tqdm(total=len(times)) as pbar: # Create progress bar

        cut_lens = np.diff([0] + times) # Get the time delta between times (audio delta to next beat [s])

        mv_clips = []

        cut_len = cut_lens[0]

        # Generate subclips from videos in video directory by splitting video on scene changes
        # Iterate through each of these clips
        for clip in clip_generator:
            clip_len = clip.duration

            # Video clip must be longer than audio split time so clip can be trimmed down to match audio len
            if clip_len > cut_len:
                mv_clips += [clip.subclip(0, cut_len)]

                # Number of clips is still less than needed to finish music video
                if len(mv_clips) < len(cut_lens):
                    cut_len = cut_lens[len(mv_clips)]
                    pbar.update(1) # Update progress bar
                else: # All clips created to match audio beats
                    return mv_clips

        return mv_clips

def get_clips(single=True, chunk_size=10):
    clips = []
    for video in tqdm(VID_FILES):
        print(f'Processing video file: {video}')
        for clip in split_video(video, check_freq=CHECK_FREQ):
            if single:
                yield clip
            else:
                clips += [clip]

    if not(single):
        print(f'{len(clips)} clips collected.')

        while True:

            new_len = (len(clips) // chunk_size) * chunk_size
            clips = clips[:new_len]

            # Create list of indices that are shuffled in chunks
            shuffle_idxs = np.arange(new_len) # Create index array
            shuffle_idxs = shuffle_idxs.reshape(-1, chunk_size) # Reshape for shuffling in chunks
            np.random.shuffle(shuffle_idxs)
            shuffle_idxs = shuffle_idxs.flatten()

            clips = [clips[idx] for idx in shuffle_idxs] # Shuffle clips with indices

            for clip in clips:
                yield clip

def get_unique_export_name(requested_name):
    new_name = requested_name
    i = 1
    # Append unique id suffix to filename if file already exists
    name = requested_name.split('.')[0]
    ext = requested_name.split('.')[-1]
    while os.path.exists(new_name):
        i += 1
        new_name = name + str(i) + '.' + ext
    return new_name

print('Getting split times from audio file...')
times = get_split_times(audio_data)

print('Building music video. This will take a long time...')

shuffle = True
if SHUFFLE_CNT == 0:
    shuffle = False
    SHUFFLE_CNT = 1
else:
    print(f'{SHUFFLE_CNT} music videos to be created with same clips shuffled on each iteration.')

clip_generator = get_clips(single=not(shuffle), chunk_size=SHUFFLE_CHUNK_SIZE)

for export_cnt in range(SHUFFLE_CNT):
    mv_clips = build_mv_clips(times, clip_generator)

    assert len(mv_clips) > 0, "Error no clips created. Clip lens may be too short for audio splice times."

    print(f'Build complete. Cut {len(mv_clips)} clips to match audio slices. Exporting video...')
    music_video = concatenate_videoclips(mv_clips)

    music_audio = AudioFileClip(FINAL_AUDIO).subclip(0, music_video.duration)

    final_music_video = music_video.set_audio(music_audio)

    mv_name = get_unique_export_name(EXPORT_FILENAME) # Appends unique index to export name

    print(f'Exporting music video file {mv_name}...')
    final_music_video.write_videofile(mv_name)

    print(f'Complete {export_cnt + 1} of {SHUFFLE_CNT} complete.')

print('Done. Total processing time took {} minutes.'.format((time.time() - start_time)/60))