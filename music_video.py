from moviepy.editor import VideoFileClip, ImageClip
from video import get_clips_from_img_dir, get_clip_times, VIDEO_EXTENSIONS, IMG_EXTENSIONS
import numpy as np
from tqdm import tqdm
import os

def build_mv_clips(video_path_list, audio_split_times, shuffle=False, use_once=False, thresh=5, thresh_inc=5, max_thresh=20):

    with tqdm(total=len(audio_split_times)) as pbar:  # Create progress bar

        audio_cut_lens = np.diff(audio_split_times)  # Get the time delta between times (audio delta to next beat [s])

        mv_clips = []

        audio_cut_len = audio_cut_lens[0]

        prev_clip_len = 0
        while thresh < max_thresh:
            for path, clip_times in get_clip_times(video_path_list, shuffle=shuffle, use_once=use_once, split_thresh=thresh):
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