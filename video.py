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

def get_clip_times(video_path_list, split_thresh=5, use_once=False, shuffle=False, frame_check_freq=1, max_time=5000, chunk_size=20):
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

            if shuffle:
                split_times = shuffle_in_chunks(split_times, chunk_size=chunk_size)
            yield path, split_times

        if use_once:
            break

def build_musicvideo_clips(video_path_list, audio_split_times, shuffle=False, use_once=False, thresh=5, thresh_inc=5, max_thresh=20, chunk_size=20):

    with tqdm(total=len(audio_split_times)) as pbar:  # Create progress bar

        audio_cut_lens = np.diff(audio_split_times)  # Get the time delta between times (audio delta to next beat [s])

        mv_clips = []

        audio_cut_len = audio_cut_lens[0]

        prev_clip_len = 0
        while thresh < max_thresh:
            for path, clip_times in get_clip_times(video_path_list, shuffle=shuffle, use_once=use_once, split_thresh=thresh, chunk_size=chunk_size):
                init_video = False

                for start_time, stop_time in clip_times:
                    clip_len = stop_time - start_time

                    # Video clip must be longer than audio split time so clip can be trimmed down to match audio len
                    if clip_len > audio_cut_len:
                        if not(init_video):
                            if path.split('.')[-1] in VIDEO_EXTENSIONS:
                                video = VideoFileClip(path)
                            elif path.split('.')[-1] in IMG_EXTENSIONS:
                                video = ImageClip(path).set_pos(("center", "center")).resize(height=1080)
                            init_video = True

                        # Add video clip to music video
                        mv_clips += [video.subclip(start_time, start_time + audio_cut_len)]

                        # Number of clips is still less than needed to finish music video
                        if len(mv_clips) < len(audio_cut_lens):
                            audio_cut_len = audio_cut_lens[len(mv_clips)]
                            pbar.update(1)  # Update progress bar
                        else:  # All clips created to match audio beats
                            return mv_clips # List filled to completion

            if (len(mv_clips) - prev_clip_len) == 0:
                print(f'No clips added using threshold {thresh}. Trying increased split threshold {thresh + thresh_inc}.')
                thresh += thresh_inc

            prev_clip_len = len(mv_clips)

        return mv_clips # Stopped short of completion