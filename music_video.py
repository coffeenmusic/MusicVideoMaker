from moviepy.editor import VideoFileClip
from video import get_clips_from_dir
import numpy as np
from tqdm import tqdm
import os

def build_mv_clips(times, clip_generator, use_clip_dir=False, shuffle=False, chunk_size=20):

    with tqdm(total=len(times)) as pbar: # Create progress bar

        cut_lens = np.diff([0] + times) # Get the time delta between times (audio delta to next beat [s])

        mv_clips = []

        cut_len = cut_lens[0]

        if use_clip_dir:
            clip_generator = get_clips_from_dir(shuffle=shuffle, chunk_size=chunk_size)

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