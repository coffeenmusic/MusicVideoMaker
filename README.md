
# Overview
Song + Video and/or Images + Python = Music Video
1. Choose a song. Currently only supports .wav Audio.
2. Choose a set of videos (or clips).
3. Generate Audio Thresholds: set_audio_thresholds.py which creates a threshold file used when processing the music video.
4. Create Music Video: run.py which splits videos on beats which exceed the threshold defined in the previous step. Then it stitches all the clips together and exports the final music video.

# Dependencies
- Uses ffmpeg, ensure ffmpeg installed and PATH links to it's location.
- MoviePy: Splices video clips and stitches them back together with audio.
- Decord: Iterates video frames (Faster than MoviePy).
- PyAudio: Reads audio data.
- Pygame: Used for setting the music's split threshold.
- Spleeter: Separates song in to drum, vocal, bass, and other tracks to use for finding split times. This allows for cleaner processing.

# :notes:Generate Audio Thresholds
    set_audio_thresholds.py

This will play the audio file in real time and display an equalizer. Each bar on the equalizer represents a frequency range. Click on a horizontal bar to set its threshold level, this will add a red horizontal bar at that frequency range (Right click to reset). You can set different thresholds for each frequency range or leave them at 0 to ignore that range. Once a threshold is set, the screen will flash blue every time that threshold is exceeded to indicate where video will be cut. Press space bar to toggle the equalizer display and only show the blue flash. After exiting the tool, the thresholds will be saved and imported automatically when creating a music video.

# Options
    run.py
    
	-a path\file.wav: Audio Reference Path & File
	-m path\file.wav: Music file
	-v path: Video Directory Path
	-n export_filename.mp4: Name & file extension of exported music video. Defaults to music_video.mp4 if not used.
	-shuffle count: Shuffles clips and exports number of music videos specified
	-start: Start music video creation at this timestamp in seconds
	-stop: Stop music video creation at this timestamp in seconds
	-use_once: Use each video clip once and then stop even if entire video not complete
	-export_clips: Uses video files in video directory and chops them in to clips
	-use_clip_dir: Uses clips in clip directory to create music video
	-freq seconds: How often in seconds to compare video frames for a scene change. Default 1 second.

# :movie_camera:Create Music Video - Examples 
Examples assume 'Media\Videos\' video directory

### Run w/ song and reference audio 
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -a "Media\Audio\drums.wav"
Reference audio is used to more easily pick out beats where video should be split. I recommend using spleeter
to split the music in drums, vocals, etc. and generally use the drums track.


	
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
- Build spleeter in to processing
- Update yaml environment file
- Create pip package
- Add arg for changing check frequency from 1 second to other values
- Add Media, Images, Video, Clips, & Audio directories as default to project and git
- [Image Segmentation](https://zulko.github.io/moviepy/examples/compo_from_image.html)

# Spleeter Command
spleeter separate -p spleeter:4stems -o output audio_example.wav
