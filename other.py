from PIL import Image
from matplotlib.pyplot import imshow, show
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