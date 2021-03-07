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