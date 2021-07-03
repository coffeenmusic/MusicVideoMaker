from other import add_dirs_if_not_exists

VID_DIR = os.path.join('Media', 'Videos') # Default video directory
CLIP_DIR = os.path.join('Media', 'Clips')
AUDIO_DIR = os.path.join('Media', 'Audio') # Default audio directory

add_dirs_if_not_exists([VID_DIR, AUDIO_DIR, CLIP_DIR])

