from PIL import Image
from matplotlib.pyplot import imshow, show

def print_frame(frame):
    img = Image.fromarray(frame)
    imshow(img)
    show()
