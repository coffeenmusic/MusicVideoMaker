from audio import get_audio_data, get_saved_audio, get_split_times, is_increasing, separate_audio_tracks, SEPARATE_DICT
from video import build_musicvideo_clips, export_clips, VIDEO_EXTENSIONS, IMG_EXTENSIONS
from other import get_unique_filename, add_dirs_if_not_exists, get_default_files
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from decord import VideoReader
from decord import cpu, gpu
import wave
import pyaudio
from tqdm import tqdm
import numpy as np
import sys
import os
import time
import pickle

SEPARATED_AUDIO_FILE = None
MUSIC_FILE = None
VID_DIR = os.path.join('Media', 'Videos') # Default video directory
CLIP_DIR = os.path.join('Media', 'Clips')
AUDIO_DIR = os.path.join('Media', 'Audio') # Default audio directory
EXPORT_FILENAME = 'music_video.mp4'
SAVED_THRESH_FILENAME = 'saved_thresholds.pkl'
SHUFFLE_CNT = 0
USE_ONCE = False
SHUFFLE_CHUNK_SIZE = 20
CHECK_FREQ = 1
CHUNK_SIZE = 20
USE_DECORD = False
EXPORT_CLIPS = False
USE_CLIP_DIR = False
START_TIME = 0
STOP_TIME = 0
args = sys.argv
INSTRUMENT = 'drums.wav'
HEIGHT = 1080

i = 0
while True:
    if args[i] in ['-v', '-video']:
        i += 1
        VID_DIR = str(args[i])
    elif args[i] in ['-a', '-audio']:
        i += 1
        SEPARATED_AUDIO_FILE = str(args[i])
    elif args[i] in ['-m', '-music']:
        i += 1
        MUSIC_FILE = str(args[i])
    elif args[i] == '-n':
        i += 1
        EXPORT_FILENAME = str(args[i])
    elif args[i] == '-instrument':
        i += 1
        INSTRUMENT = SEPARATE_DICT[int(args[i])]
    elif args[i] == '-use_decord':
        USE_DECORD = True
    elif args[i] == '-shuffle':
        i += 1
        SHUFFLE_CNT = int(args[i])
    elif args[i] == '-freq':
        i += 1
        CHECK_FREQ = float(args[i])
    elif args[i] == '-start':
        i += 1
        START_TIME = float(args[i])
    elif args[i] == '-stop':
        i += 1
        STOP_TIME = float(args[i])
    elif args[i] == '-height':
        i += 1
        HEIGHT = int(args[i])
    elif args[i] == '-chunk':
        i += 1
        CHUNK_SIZE = float(args[i])
    elif args[i] == '-export_clips':
        EXPORT_CLIPS = True
    elif args[i] == '-use_clip_dir':
        USE_CLIP_DIR = True
    elif args[i] == '-use_once': # Use each clip only once
        USE_ONCE = True
    elif i != 0:
        print(f'Command argument {args[i]} not recognized.')
        exit(0)

    i += 1
    if i >= len(args):
        break
        
if not MUSIC_FILE:
    print('No song provided by user. For user specified song use -m command argument.')
    files = get_default_files(os.path.join('Media', 'Audio'), ('.mp3', '.wav'))
    if files != None:
        MUSIC_FILE = files[0]
        print(f'Music file found: {MUSIC_FILE}. To use another song, run command with -music filename.mp3')
        
if not MUSIC_FILE:
    print('Auto Music Video Maker Commands:')
    print('-a Audio\\Path\\ref_audio.wav')
    print('-m Audio\\Path\\final_music.wav')
    print('-v Video\\Set\\Dir\\')
    print('-n export_filename.mp4')
    exit(0)

add_dirs_if_not_exists([VID_DIR, AUDIO_DIR, CLIP_DIR])

VIDEO_FILES = [os.path.join(VID_DIR, f) for f in os.listdir(VID_DIR) if f.split('.')[-1].lower() in VIDEO_EXTENSIONS]
assert len(VIDEO_FILES) > 0, f'No videos found in video directory {VID_DIR}'

if EXPORT_CLIPS:
    export_clips(VIDEO_FILES, clip_dir=CLIP_DIR)
    exit(0)

if not SEPARATED_AUDIO_FILE:
    save_dir = separate_audio_tracks(MUSIC_FILE)
    SEPARATED_AUDIO_FILE = os.path.join(save_dir, INSTRUMENT)

# Verify audio files exist
assert os.path.exists(SEPARATED_AUDIO_FILE), f'Audio file {SEPARATED_AUDIO_FILE} not found.'
assert os.path.exists(MUSIC_FILE), f'Audio file {MUSIC_FILE} not found.'

print('Video Directory: ', VID_DIR)
print('Reference Audio File: ', SEPARATED_AUDIO_FILE)
print('Song Used in Final Music Video: ', MUSIC_FILE)
print('Export Filename: ', EXPORT_FILENAME)

start_timer = time.time()

saved_data = get_saved_audio(SEPARATED_AUDIO_FILE)
if saved_data:
    audio_data, CHUNK, RATE = saved_data
else:
    audio_data, CHUNK, RATE = get_audio_data(SEPARATED_AUDIO_FILE)

# Import saved audio amplitude threshold data
thresh_path = os.path.join(os.path.dirname(SEPARATED_AUDIO_FILE), SAVED_THRESH_FILENAME)
saved_thresholds = pickle.load(open(thresh_path, "rb"))
audio_thresholds = saved_thresholds['thresholds']
freq_buckets = saved_thresholds['buckets']
freq_buckets_min = saved_thresholds['min_buckets']
freq_buckets_max = saved_thresholds['max_buckets']

STOP_TIME = len(audio_data)*(CHUNK/RATE) if STOP_TIME == 0 else STOP_TIME
print(f'Audio to be processed between {START_TIME}s & {STOP_TIME}s')
print('Getting split times from audio file...')
audio_split_times = get_split_times(audio_data, RATE, audio_thresholds, freq_buckets, freq_buckets_min, freq_buckets_max, chunk=CHUNK, start_time=START_TIME, stop_time=STOP_TIME)
print(f'{len(audio_split_times)} audio slices created.')

print('Building music video. This will take a long time...')

shuffle = True
if SHUFFLE_CNT == 0:
    shuffle = False
    SHUFFLE_CNT = 1
else:
    print(f'{SHUFFLE_CNT} music videos to be created with same clips shuffled on each iteration.')

if USE_CLIP_DIR:
    VIDEO_FILES = [os.path.join(CLIP_DIR, d) for d in os.listdir(CLIP_DIR) if d.split('.')[-1] in VIDEO_EXTENSIONS + IMG_EXTENSIONS]

for export_cnt in range(SHUFFLE_CNT):
    mv_clips = build_musicvideo_clips(VIDEO_FILES, audio_split_times, shuffle=shuffle, chunk_size=CHUNK_SIZE, video_height=HEIGHT)
    assert len(mv_clips) > 0, "Error no clips created. Clip lens may be too short for audio splice times."

    print(f'Build complete. Cut {len(mv_clips)} clips to match audio slices. Exporting video...')
    music_video = concatenate_videoclips(mv_clips, method='compose')

    STOP_TIME = audio_split_times[-1] if music_video.duration < STOP_TIME or STOP_TIME == 0 else STOP_TIME
    music_audio = AudioFileClip(MUSIC_FILE).subclip(START_TIME, STOP_TIME)

    final_music_video = music_video.set_audio(music_audio)

    mv_name = get_unique_filename(EXPORT_FILENAME) # Appends unique index to export name

    print(f'Exporting music video file {mv_name}...')
    fps = music_video.fps
    if not(fps):
        fps = 30
    final_music_video.write_videofile(mv_name, fps=fps)

    print(f'Complete {export_cnt + 1} of {SHUFFLE_CNT} complete.')

print('Done. Total processing time took {} minutes.'.format((time.time() - start_timer)/60))