from audio import open_stream, get_audio_data, get_saved_audio, fft_to_buckets, get_minmax_bucket_freq
import os
import numpy as np
import pygame
from pygame import Color, surfarray
import pickle
import sys

AUD_DIR = os.path.join('Media', 'Audio')
AUD_FILE = os.path.join(AUD_DIR, 'drums.wav')
SAVE_FILE = 'saved_thresholds.pkl'
TEST_THRESHOLDS = False

i = 0
args = sys.argv
while True:
    if args[i] == '-a':
        i += 1
        AUD_FILE = str(args[i])
    elif i != 0:
        print(f'Command argument {args[i]} not recognized.')
        exit(0)

    i += 1
    if i >= len(args):
        break

if not(os.path.exists(AUD_FILE)):
    print('Audio filepath cannot be found.')
    exit(0)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

cell_size = 10 # Number of pixels per cell

cell_width = SCREEN_WIDTH//cell_size
cell_height = SCREEN_HEIGHT//cell_size

display_width = cell_width*cell_size
display_height = cell_height*cell_size

saved_data = get_saved_audio(AUD_FILE)
if saved_data:
    audio_data, CHUNK, RATE = saved_data
else:
    audio_data, CHUNK, RATE = get_audio_data(AUD_FILE)

buckets = [31.25 * 2 ** (n) for n in range(10)]


min_bucket, max_bucket = get_minmax_bucket_freq(audio_data, buckets, RATE)


def new_state(freq_buckets, min_bucket, max_bucket):
    state = np.zeros((cell_height, cell_width))

    bucket_px_width = int(cell_width / len(freq_buckets))

    # Scale buckets
    scaled = (freq_buckets - min_bucket) / (max_bucket - min_bucket)
    multiplier = 1
    scaled = [int(np.round(s * cell_height * multiplier)) for s in scaled for _ in range(bucket_px_width)]
    scaled = [s if s < cell_height else cell_height - 1 for s in scaled]

    for i, s in enumerate(scaled):
        state[i, :s] = 1

    state = np.flip(state, axis=1)
    return state


def state_to_px(state, px, thresh, buckets):
    temp = np.where(state == 1, 0, 255)
    for i in range(px.shape[-1]):
        px[:, :, i] = temp

    scaled = (buckets - min_bucket) / (max_bucket - min_bucket)

    for i in range(len(buckets)):
        idx_thresh = thresh[i]

        bucket_width = int(cell_width / len(buckets))
        xpos = int(i * bucket_width)
        px[xpos:xpos + bucket_width, int((1 - idx_thresh) * px.shape[0]) - 1, 0] = 255
        px[xpos:xpos + bucket_width, int((1 - idx_thresh) * px.shape[0]) - 1, 1] = 0
        px[xpos:xpos + bucket_width, int((1 - idx_thresh) * px.shape[0]) - 1, 2] = 0

        if idx_thresh > 0 and scaled[i] > idx_thresh:
            px[:, :, 0] = 0
            px[:, :, 1] = 0
            px[:, :, 2] = 255
    return px


stream, wf, CHUNK = open_stream(AUD_FILE)

# Initialize display state
state = np.zeros((cell_height, cell_width))
pygame.init()
screen = pygame.display.set_mode((display_height, display_width))
pygame.display.set_caption('Threshold Helper')

# create a surface with the size as the array
surface = pygame.Surface((cell_height, cell_width))
surface = pygame.transform.scale(surface, (display_height, display_width))  # Scale Size Up

# Initialize screen to white
px = pygame.surfarray.pixels3d(surface)
px[:, :, :] = np.ones((display_height, display_width, 3)) * 255

run = True
cnt = 0
thresh = {i: 0 for i in range(len(buckets))}
while run:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left Click
                pos = pygame.mouse.get_pos()
                pct_height = 1 - (pos[1] / display_height)
                pct_width = pos[0] / display_width
                click_idx = int(np.round((len(buckets) - 1) * pct_width))

                # Save audio threshold for frequency range of the index clicked
                thresh[click_idx] = pct_height
            elif event.button == 3: # Right Click
                pos = pygame.mouse.get_pos()
                pct_width = pos[0] / display_width
                click_idx = int(np.round((len(buckets) - 1) * pct_width))
                thresh[click_idx] = 0
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                TEST_THRESHOLDS = not(TEST_THRESHOLDS)

        if event.type == pygame.QUIT:
            run = False

    # Scale Screen Size Down to edit pixel array at cell size=1 --------------------------------------------------
    surface = pygame.transform.scale(surface, (cell_height, cell_width))
    px = pygame.surfarray.pixels3d(surface)

    data = audio_data[cnt]

    stream.write(bytes(data))

    n = len(data)
    fhat = np.fft.fft(data, n)
    PSD = np.abs(fhat * np.conj(fhat) / n)  # Power Spectral Density
    freq = (RATE / n) * np.arange(n)

    fb, idxs = fft_to_buckets(freq, PSD, buckets)  # Chunk frequencies in to buckets

    # Iterate State
    if TEST_THRESHOLDS:
        state = np.zeros((cell_height, cell_width))
    else:
        # Display equalizer
        state = new_state(fb, min_bucket, max_bucket)
    px = state_to_px(state, px, thresh, fb)

    # Scale Screen Size Up to display pixel array at cell size=multiplier ----------------------------------------
    surface = pygame.transform.scale(surface, (display_height, display_width))  # Scale Size Up
    screen.blit(surface, (0, 0))  # Update all pixels on screen object

    pygame.display.update()

    cnt += 1

pygame.quit()

# Save audio thresholds to pickle file
pickle.dump({'thresholds': thresh,
             'buckets': buckets,
             'min_buckets': min_bucket,
             'max_buckets': max_bucket,
             'audio_file': AUD_FILE.split('\\')[-1]}, open(os.path.join(AUD_DIR, SAVE_FILE), "wb"))