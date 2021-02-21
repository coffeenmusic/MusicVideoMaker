from moviepy.editor import VideoFileClip
from decord import VideoReader
from decord import cpu, gpu
import cv2
import numpy as np
from tqdm import tqdm

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