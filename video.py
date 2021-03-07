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

def get_video_split_times(vid_filename, check_freq=1, split_thresh=10):
    """
    check_freq [seconds] - how often to compare two frames for scene change
    split_thresh - mean difference in pixel values allowed before triggering split
    """
    vr = VideoReader(vid_filename, ctx=cpu(0))
    fps = vr.get_avg_fps()

    start_time = 0  # time in seconds from video where current clip starts

    frame_freq = int(fps * check_freq)

    times = []
    for i in range(0, len(vr), frame_freq):
        frame = vr[i].asnumpy()

        if i > 0:  # Skip first frame
            if start_time != stop_time and scene_changed(prev_frame, frame, delta_thresh=split_thresh):
                times += [(start_time, stop_time)]

                start_time = i/fps

        prev_frame = frame
        stop_time = i/fps

    if len(times) == 0:
        times += [(0, stop_time)]

    return times

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
    if len(in_list) == 1:
        return in_list

    chunk_size = int(len(in_list) / 2) if chunk_size > len(in_list) / 2 else chunk_size # Allow minimum shuffle if list too small or chunk too large
    new_len = (len(in_list) // chunk_size) * chunk_size
    in_list = in_list[:new_len]

    # Create list of indices that are shuffled in chunks
    shuffle_idxs = np.arange(new_len)  # Create index array
    shuffle_idxs = shuffle_idxs.reshape(-1, chunk_size)  # Reshape for shuffling in chunks
    np.random.shuffle(shuffle_idxs)
    shuffle_idxs = shuffle_idxs.flatten()

    return [in_list[i] for i in shuffle_idxs]

def get_clip_times(video_path_list, split_thresh=5, use_once=False, shuffle=False, frame_check_freq=1, max_time=5000):
    """
    Iterate video frames, split at scene changes, and create clips to yield back
        video_path_list - a list of paths to all videos being iterated on
        shuffle - shuffle clips if True else use in order they are listed
        frame_check_freq - how often in seconds to compare frames for scene change
    """
    assert len(video_path_list) > 0, "Empty video path list."

    while True:
        video_path_list = shuffle_in_chunks(video_path_list, chunk_size=1) if shuffle else video_path_list
        for video_cnt, path in enumerate(video_path_list):
            if path.split('.')[-1] in VIDEO_EXTENSIONS:
                split_times = get_video_split_times(path, check_freq=frame_check_freq, split_thresh=split_thresh)
            elif path.split('.')[-1] in IMG_EXTENSIONS:
                split_times = [(0, max_time)]
            yield path, split_times

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