from video import split_video, split_video_decord
import numpy as np
from tqdm import tqdm

def get_clips(video_path_list, single=True, chunk_size=20, frame_check_freq=1):
    """
    video_path_list - a list of paths to all videos being iterated on
    single - run through clips one time without shuffling
    chunk_size - number of clips to keep unshuffled when shuffling all clips
    frame_check_freq - how often in seconds to compare frames for scene change
    """

    clips = []
    for video_cnt, video in enumerate(video_path_list):
        print(f'Processing video file {video_cnt}/{len(video_path_list)}: {video}')
        for clip in split_video(video, check_freq=frame_check_freq):
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