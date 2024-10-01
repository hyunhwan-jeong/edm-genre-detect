import numpy as np
import soundfile as sf
from transformers import pipeline

# Load your WAV file
file_path = "/Volumes/T9/TUNES/2katz/Always.wav"
audio, samplerate = sf.read(file_path)

# Check if audio has more than one channel and convert to mono by averaging the channels
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Ensure the audio is at 16kHz
if samplerate != 16000:
    from scipy.signal import resample
    audio = resample(audio, int(16000 * len(audio) / samplerate))

# Initialize the pipeline with the audio classification model
pipe = pipeline("audio-classification", model="mtg-upf/discogs-maest-30s-pw-129e", trust_remote_code=True)

# Perform classification
results = pipe(audio)
print(results)

