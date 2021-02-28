Currently only supports .wav Audio

# Options
  - a path\file.wav: Audio Reference Path & File
  - m path\file.wav: Music file
  - v path: Video Directory Path
  - n export_filename.mp4: Name & file extension of exported music video. Defaults to music_video.mp4 if not used.
  - t value: threshold value if changed from default
  - shuffle count: Shuffles clips and exports number of music videos specified
  - start/stop seconds: Number of seconds of start and/or stop times
  - use_once: Use each video clip once and then stop even if entire video not complete
  - use_img_dir: Use Media\Images\ directory and images in that directory to build video
  - export_clips
  - use_clip_dir:
  - freq seconds: How often in seconds to compare video frames for a scene change. Default 1 second.

# Examples 
Examples assume 'Media\Videos\' video directory

### Run w/ song and reference audio 
Reference audio is used to more easily pick out beats where video should be split. I recommend using spleeter
to split the music in drums, vocals, etc. and generally use the drums track.
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav"
	
### Export video to clips directory as separate clips
	run.py -export_clips

### Use clips directory to create music video
Note: Unwanted clips should be removed from this directory before processing
	run.py -use_clip_dir
	
### Create music video from audio between start & stop time [seconds]
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -start 20 -stop 40
	
### Create 5 music videos w/ shuffled clips
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -shuffle 5
	
### Only use videos from video directory once & stop when out of video clips
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -use_once
	
### Create video from images
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -use_img_dir
	
### Use different video directory
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -v C:\My\Video\Directory\
	
### Change audio splice amplitude threshold
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav" -t 500000000

# TODO:
- Create thresh pickle file from threshold tool
- Build spleeter in to processing
- [Image Segmentation](https://zulko.github.io/moviepy/examples/compo_from_image.html)

# Spleeter Command
spleeter separate -p spleeter:4stems -o output audio_example.wav
