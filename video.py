from moviepy.editor import VideoFileClip, ImageClip
from decord import VideoReader
from decord import cpu, gpu
import cv2
import numpy as np
from tqdm import tqdm
import os
import psutil

VIDEO_EXTENSIONS = ['mp4', 'avi', 'mkv', 'm4v']
IMG_EXTENSIONS = ['jpg', 'jpeg'] #, 'png', 'bmp', 'gif', 'tif'

def scene_changed(prev_frame, frame, delta_thresh=10):
    delta = abs(np.mean(prev_frame) - np.mean(frame))

    if delta > delta_thresh:
        return True
    return False

def split_video_decord(vid_filename, check_freq=1, split_thresh=10):
    """
    check_freq [seconds] - how often to compare two frames for scene change
    split_thresh - mean difference in pixel values allowed before triggering split
    """
    with open(vid_filename, 'rb') as f:
        video = VideoFileClip(vid_filename)
        vr = VideoReader(vid_filename, ctx=cpu(0))

        clip_cnt = 0  # Number of clips created from video
        start_time = 0  # time in seconds from video where current clip starts

        frame_freq = int(video.reader.fps * check_freq)
        print(f'Compare frames every {check_freq} seconds. This equals {frame_freq} frames.')

        total_frames = len(vr)

        prev_frame = vr[0]

        no_change_cnt = 0
        for i in range(0, total_frames, frame_freq):
            frame = vr[i].asnumpy()

            if i > 0:  # Skip first frame
                if start_time != stop_time and scene_changed(prev_frame, frame, delta_thresh=split_thresh):
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

def split_video(vid_filename, check_freq=1, split_thresh=10):
    """
    check_freq [seconds] - how often to compare two frames for scene change
    split_thresh - mean difference in pixel values allowed before triggering split
    """
    clip_cnt = 0  # Number of clips created from video
    start_time = 0  # time in seconds from video where current clip starts
    video = VideoFileClip(vid_filename)

    frame_freq = int(video.reader.fps * check_freq)
    print(f'Compare frames every {check_freq} seconds. This equals {frame_freq} frames.')

    prev_frame = video.get_frame(0)  # Initialize previous frame

    for i, (time, frame) in tqdm(enumerate(video.iter_frames(with_times=True))):

        if i % frame_freq == 0:
            if i > 0:  # Skip first frame
                if start_time != stop_time and scene_changed(prev_frame, frame, delta_thresh=split_thresh):
                    yield video.subclip(start_time, stop_time)

                    start_time = time

                    clip_cnt += 1

            prev_frame = frame.copy()
            stop_time = time

def export_clips(clip_generator, path=None):
    if path == None:
        path = os.path.join('Media', 'Clips')

    if not (os.path.exists(path)):
        os.mkdir(path)

    for clip in clip_generator:
        files = os.listdir(path)

        largest_id = 0
        ids = [int(f.split('.')[0]) for f in files if f.split('.')[-1] in VIDEO_EXTENSIONS and f.split('.')[0].isdigit()]
        if len(ids) > 0:
            largest_id = max(ids)


        idx = largest_id + 1
        clip_name = str(idx) + '.mp4'
        while os.path.exists(os.path.join(path, clip_name)):
            idx += 1
            clip_name = str(idx) + '.mp4'

        clip.write_videofile(os.path.join(path, clip_name), verbose=False)

def shuffle_in_chunks(in_list, chunk_size=20):
    """
    Shuffle a list but group together list items close to eachother in chunks
        in_list - list to be shuffled
        chunk_size - length of clips grouped together that aren't shuffled (Example: [3,4,1,2,5,6] chunk=2
    """
    chunk_size = int(len(in_list) / 2) if chunk_size > len(in_list) / 2 else chunk_size # Allow minimum shuffle if list too small or chunk too large

    new_len = (len(in_list) // chunk_size) * chunk_size
    in_list = in_list[:new_len]

    # Create list of indices that are shuffled in chunks
    shuffle_idxs = np.arange(new_len)  # Create index array
    shuffle_idxs = shuffle_idxs.reshape(-1, chunk_size)  # Reshape for shuffling in chunks
    np.random.shuffle(shuffle_idxs)
    shuffle_idxs = shuffle_idxs.flatten()

    return [in_list[i] for i in shuffle_idxs]

def get_clips(video_path_list, use_once=True, shuffle=False, chunk_size=10, frame_check_freq=1, use_decord=False, max_clips=100):
    """
    Iterate video frames, split at scene changes, and create clips to yield back
        video_path_list - a list of paths to all videos being iterated on
        use_once - run through clips one time without shuffling
        shuffle - shuffle clips if True else use in order they are listed
        chunk_size - number of clips to keep unshuffled when shuffling all clips
        frame_check_freq - how often in seconds to compare frames for scene change
        use decord - decord is faster than moviepy, but seems to be more buggy and doesn't alway work
        max_clips - creates list of clips with this size so those clips can be shuffled. Otherwise shuffle wouldn't be possible
    """
    max_clips = len(video_path_list) if len(video_path_list) < max_clips else max_clips

    while True:
        for video_cnt, path in enumerate(video_path_list):
            split_generator = split_video_decord(path, check_freq=frame_check_freq) if use_decord else split_video(path, check_freq=frame_check_freq)

            print(f'Processing video file {video_cnt + 1}/{len(video_path_list)}: {path}')
            clips_available = True
            while clips_available:
                # Collect clips in chunks of max_clips
                for _ in range(max_clips):
                    try:
                        clip = next(split_generator)
                    except:
                        clips_available = False

                    clips += [clip]

                clips = shuffle_in_chunks(clips, chunk_size=chunk_size) if shuffle else clips
                for c in clips:
                    yield c

                clips = []

        if use_once:
            break

def get_clips_from_dir(path=None, use_once=False, shuffle=False, chunk_size=20):
    """
    Get clips from a clip directory:
        path - path to the clip dir that contains video clips
        use_once - run through clips one time without reusing clips
        shuffle - shuffle clips if True else use in order they are listed
        chunk_size - number of clips to keep unshuffled when shuffling all clips
    """
    if path == None:
        path = os.path.join('Media', 'Clips')

    path_list = [os.path.join(path, d) for d in os.listdir(path) if d.split('.')[-1] in VIDEO_EXTENSIONS]

    while True:
        clip_paths = shuffle_in_chunks(path_list, chunk_size=chunk_size) if shuffle else path_list
        for p in clip_paths:
            yield VideoFileClip(p)

        if use_once:
            break

def get_clips_from_img_dir(path=None, use_once=False, shuffle=False, chunk_size=20, height=1080):
    """
    Get images from an image directory, convert them to video clips and yield back:
        path - path to the image dir that contains images
        use_once - run through images one time without reusing images
        shuffle - shuffle images if True else use in order they are listed
        chunk_size - number of images to keep unshuffled when shuffling all images
        height - resize all images with this height. Ratio for each individual image should remain the same
    """

    if path == None:
        if os.path.exists(os.path.join('Media', 'Images')):
            path = os.path.join('Media', 'Images')
        elif os.path.exists(os.path.join('Media', 'Videos')):
            path = os.path.join('Media', 'Videos')
        else:
            path = ''

    assert os.path.exists(path), "Image directory not found."

    path_list = [os.path.join(path, d) for d in os.listdir(path) if d.split('.')[-1] in IMG_EXTENSIONS]

    while True:
        img_paths = shuffle_in_chunks(path_list, chunk_size=chunk_size) if shuffle else path_list
        for p in img_paths:
            #yield ImageClip(p).set_duration(2).set_pos(("center", "center")).resize(height=height)
            try:
                img = ImageClip(p).set_pos(("center", "center")).resize(height=height)
            except:
                continue

            yield img

        if use_once:
            break