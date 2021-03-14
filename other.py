from PIL import Image
from matplotlib.pyplot import imshow, show
import numpy as np
import os

def print_frame(frame):
    img = Image.fromarray(frame)
    imshow(img)
    show()

def get_unique_filename(requested_name):
    new_name = requested_name
    i = 1
    # Append unique id suffix to filename if file already exists
    name = requested_name.split('.')[0]
    ext = requested_name.split('.')[-1]
    while os.path.exists(new_name):
        i += 1
        new_name = name + str(i) + '.' + ext
    return new_name
    
def add_dirs_if_not_exists(dir_list):
    for dir in dir_list:
        if not(os.path.exists(dir)):
            os.mkdir(dir)

def get_next_path_index(path, ext_list=None):
    next_idx = 0

    if ext_list: # Only extensions from extension list
        ids = [int(f.split('.')[0]) for f in os.listdir(path) if f.split('.')[-1] in ext_list and f.split('.')[0].isdigit()]
    else: # All extensions valid
        ids = [int(f.split('.')[0]) for f in os.listdir(path) if f.split('.')[0].isdigit()]

    if len(ids) > 0:
        next_idx = max(ids) + 1

    return next_idx

def get_ext(path, include_period=False):
    ext = os.path.splitext(path)[-1].lower()
    return ext if include_period else ext[1:]

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