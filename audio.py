import pyaudio
import wave
import numpy as np
from sklearn.preprocessing import Normalizer

class AudioProcessing:
    CHUNK = 1024*4
    SCALING_FACTOR = 255
    FIFO_LONG_LEN = 20 # Rolling delta's long FIFO length
    FIFO_SHORT_LEN = 1 # Rolling delta's short FIFO length

    def __init__(self, audio_file, buckets):
        assert audio_file.endswith('.wav'), 'Currenty only WAV files are supported. Extension does not contain .wav'
        self.audio_file = audio_file
        
        self.buckets = buckets
        
    def run_preprocessing(self):
        self.open_stream()
        
        assert self.CHUNK % 1024 == 0, 'CHUNK must be a multiple of 1024'
        
        all_data, all_rd = self._preprocess_audio()
        
        self.close_stream()
        
        # Normalize returned audio file data
        self.normalized_data = Normalizer().fit_transform(all_data)*self.SCALING_FACTOR # Normalize to entire track and not to individual samples
        self.rolling_deltas = all_rd
        
    def open_stream(self):
        self.wf = wave.open(self.audio_file, 'rb')
        self.RATE = self.wf.getframerate()
        self.FPS = self.RATE / self.CHUNK
        self.seconds_audio = self.wf.getnframes() / self.wf.getframerate()

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                        channels=self.wf.getnchannels(),
                        rate=self.RATE,
                        output=True)
        
    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
    def get_roll_delta(self, fifo_long, fifo_short, lfb):   
        """
        Gets the difference between a long FIFO and a short FIFO of the audio's frequency data
        """
    
        fifo_long = np.roll(fifo_long, 1, axis=0)
        fifo_long[0,:] = lfb
        roll_long = fifo_long.mean(axis=0)
        roll_long[(roll_long == float('-inf')) | (roll_long == float('inf'))] = 0.01
        
        fifo_short = np.roll(fifo_short, 1, axis=0)
        fifo_short[0,:] = lfb
        roll_short = fifo_short.max(axis=0)
        roll_short[(roll_short == float('-inf')) | (roll_short == float('inf'))] = 0.001
        
        delta = roll_short - roll_long
        delta = np.where(delta > 0, delta, 0.001)
        
        return delta, fifo_long, fifo_short

    def print_stats(self):
        print(f'framerate: {self.RATE}')
        print(f'FPS: {self.FPS}')
        print(f'Track Length: {self.seconds_audio}')
        
    def _preprocess_audio(self):
        """
        Gets each chunk from audio_file, takes the fft, then breaks the frequencies in to buckets.
        Also gets the rolling delta for each frequency bucket
        """
        data = self.wf.readframes(self.CHUNK)
        data_int = np.frombuffer(data, dtype=np.int32) # Read bytes to int
     
        fifo_long = np.zeros((self.FIFO_LONG_LEN, len(self.buckets)))
        fifo_short = np.zeros((self.FIFO_SHORT_LEN, len(self.buckets)))
        
        all_rd = np.empty((0, len(self.buckets))) # Rolling delta for every CHUNK processed. Starts empty and grows with each iteration
        cnt = 0
        while data != '':                
            freq, PSD = self._get_fft(data_int) # returns frequency and power spectral density

            # Create buckets for each frequency specified in buckets
            fb, idxs = self._fft_to_buckets(freq, PSD, self.buckets)
            
            # Log of frequencies
            lfb = np.log(fb)
            if all([f == 0 for f in fb]):
                lfb = [-1]*len(fb) 
            lfb = np.expand_dims(lfb, axis=0)
            
            roll_delta, fifo_long, fifo_short = self.get_roll_delta(fifo_long, fifo_short, lfb)
            
            # Append current rolling delta
            all_rd = np.append(all_rd, np.expand_dims(np.array(roll_delta), 0), axis=0)
            
            if cnt == 0:
                all_lfb = lfb.copy()
            else:
                all_lfb = np.append(all_lfb, lfb, axis=0)
            
            # Read next frame
            data = self.wf.readframes(self.CHUNK)
            if len(data) < self.CHUNK:
                break
        
            data_int = np.frombuffer(data, dtype=np.int32) # Read bytes to int
            data_int = np.resize(data_int, self.CHUNK) # Handle final CHUNK where size might be less than CHUNK size

            cnt += 1    
        return all_lfb, all_rd
        
    def _get_fft(self, data_int):
        n = len(data_int) 
        fhat = np.fft.fft(data_int, n)
        PSD = np.abs(fhat * np.conj(fhat) / n) # Power Spectral Density
        freq = (self.RATE / n) * np.arange(n)
        
        return freq, PSD

    def _fft_to_buckets(self, freq, PSD, buckets):
        """
        Takes the current CHUNK's frequency response and breaks each frequency in to buckets
        - freq: audio files CHUNK of data amplitudes converted in to frequencies
        - PSD: power spectral density of each frequency
        - buckets: a list of frequencies where each freq in the list will create a range between that freq and the previous
        """
        idxs = sorted({np.abs(freq - i).argmin() for i in buckets}) # Get indices of freq from closest frequencies in buckets
        
        # Average PSD values in between frequencies defined by buckets
        freq_bucket = [PSD[idxs[i]:idxs[i+1]].mean() for i in range(len(idxs)-1)]  + [PSD[idxs[-1]:].mean()] 
        
        return freq_bucket, idxs

    def init_fifo_from_preprocessed_data(self, data, FIFO_LEN):
        fifo = np.zeros((FIFO_LEN, len(self.buckets)))
        for i in range(FIFO_LEN):
            lfb = data[i]

            fifo = np.roll(fifo, 1, axis=0)
            fifo[0, :] = lfb
        return fifo
    


