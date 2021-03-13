

# Overview
Song + Video + Python = Music Video
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
    python set_audio_thresholds.py

This will play the audio file in real time and display an equalizer. Each bar on the equalizer represents a frequency range. Click on a horizontal bar to set its threshold level, this will add a red horizontal bar at that frequency range (Right click to reset). You can set different thresholds for each frequency range or leave them at 0 to ignore that range. Once a threshold is set, the screen will flash blue every time that threshold is exceeded to indicate where video will be cut. Press space bar to toggle the equalizer display and only show the blue flash. After exiting the tool, the thresholds will be saved and imported automatically when creating a music video.

# :movie_camera:Getting Started
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav"
Video files will be referenced from the video directory at Media\Videos\ or can be manually set with -v option.

# Options
    run.py
 
	-m path\file.wav: Music file
	-v path: Video Directory Path
	-n export_filename.mp4: Name & file extension of exported music video. Defaults to music_video.mp4 if not used.
	-a path\file.wav: Audio Reference Path & File (Only use to bypass spleeter and default use of separated audio file)
	-shuffle count: Shuffles clips and exports number of music videos specified
	-start: Start music video creation at this timestamp in seconds
	-stop: Stop music video creation at this timestamp in seconds
	-use_once: Use each video clip once and then stop even if entire video not complete
	-export_clips: Uses video files in video directory and chops them in to clips
	-use_clip_dir: Uses clips in clip directory to create music video
	-freq seconds: How often in seconds to compare video frames for a scene change. Default 1 second.

# Option Examples 
Examples assume 'Media\Videos\' video directory
	
### Export video to clips directory as separate clips
	run.py -export_clips
Exports to Media\Clips\

### Use clips directory to create music video
    run.py -use_clip_dir -m "Media\Audio\Greydon Square - Society Versus Nature.wav"
Note: Unwanted clips should be removed from this directory before processing
	
### Create music video from audio between start & stop time [seconds]
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -start 20 -stop 40
	
### Create 5 music videos w/ shuffled clips
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -shuffle 5
	
### Only use videos from video directory once & stop when out of video clips
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -use_once
	
### Create video from images
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav"
Run normally, but add images to Videos directory.
	
### Use different video directory
	run.py -m "Media\Audio\Greydon Square - Society Versus Nature.wav" -v C:\My\Video\Directory\

# TODO:
- Update yaml environment file
- Create pip package
- Add arg for changing check frequency from 1 second to other values
- Add Media, Images, Video, Clips, & Audio directories as default to project and git
- Make Media all directory references that are hard coded, global references in a file
- Try decord gpu with set CUDA_VISIBLE_DEVICES=1
- Add spleeter separation to set thresholds function
- If only audio filename is given, assume Media\Audio\ directory path
- [Image Segmentation](https://zulko.github.io/moviepy/examples/compo_from_image.html)

# Spleeter Command
spleeter separate -p spleeter:4stems -o output audio_example.wav
