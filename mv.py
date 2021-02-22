from audio import get_audio_data, get_saved_audio, get_split_times, is_increasing
from video import split_video, get_clips, export_clips, VIDEO_EXTENSIONS
from other import get_unique_filename
from music_video import build_mv_clips
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
EXPORT_CLIPS = False
USE_CLIP_DIR = False
amp_thresh = 1000000000
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
    elif args[i] == '-export_clips':
        EXPORT_CLIPS = True
    elif args[i] == '-use_clip_dir':
        USE_CLIP_DIR = True

    i += 1
    if i >= len(args):
        break

VID_FILES = [os.path.join(VID_DIR, f) for f in os.listdir(VID_DIR) if f.split('.')[-1].lower() in VIDEO_EXTENSIONS]

print('Video Directory: ', VID_DIR)
print('Reference Audio File: ', AUD_FILE)
print('Song Used in Final Music Video: ', FINAL_AUDIO)
print('Export Filename: ', EXPORT_FILENAME)

start_time = time.time()

saved_data = get_saved_audio(AUD_FILE)
if saved_data:
    audio_data, CHUNK, RATE = saved_data
else:
    audio_data, CHUNK, RATE = get_audio_data(AUD_FILE)

print('Getting split times from audio file...')
times = get_split_times(audio_data, RATE, amp_thresh, chunk=CHUNK)
print(f'{len(times)} audio slices created.')

print('Building music video. This will take a long time...')

shuffle = True
if SHUFFLE_CNT == 0:
    shuffle = False
    SHUFFLE_CNT = 1
else:
    print(f'{SHUFFLE_CNT} music videos to be created with same clips shuffled on each iteration.')

clip_generator = get_clips(VID_FILES, single=not(shuffle), chunk_size=SHUFFLE_CHUNK_SIZE, frame_check_freq=CHECK_FREQ)

if EXPORT_CLIPS:
    export_clips(clip_generator)
    cont = input('Clips exported. Would you like to continue [y/n]?')
    if cont.lower() in ['y', 'yes']:
        pass
    else:
        exit(0)

for export_cnt in range(SHUFFLE_CNT):
    mv_clips = build_mv_clips(times, clip_generator, use_clip_dir=USE_CLIP_DIR, shuffle=shuffle, chunk_size=SHUFFLE_CHUNK_SIZE)

    assert len(mv_clips) > 0, "Error no clips created. Clip lens may be too short for audio splice times."

    print(f'Build complete. Cut {len(mv_clips)} clips to match audio slices. Exporting video...')
    music_video = concatenate_videoclips(mv_clips)

    music_audio = AudioFileClip(FINAL_AUDIO).subclip(0, music_video.duration)

    final_music_video = music_video.set_audio(music_audio)

    mv_name = get_unique_filename(EXPORT_FILENAME) # Appends unique index to export name

    print(f'Exporting music video file {mv_name}...')
    final_music_video.write_videofile(mv_name)

    print(f'Complete {export_cnt + 1} of {SHUFFLE_CNT} complete.')

print('Done. Total processing time took {} minutes.'.format((time.time() - start_time)/60))