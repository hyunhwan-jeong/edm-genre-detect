import argparse
import json
import numpy as np
import soundfile as sf
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Audio Classification Tool')
    parser.add_argument('--input', type=str, required=True, help='Path to input WAV file')
    parser.add_argument('--output', type=str, required=False, help='Path to output JSON file')

    # Parse arguments
    args = parser.parse_args()

    # Load the WAV file
    try:
        audio, samplerate = sf.read(args.input)
    except Exception as e:
        raise SystemExit(f"Failed to read the audio file: {args.input}. Error: {str(e)}")

    # Convert to mono if necessary
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 16kHz if necessary
    if samplerate != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(16000 * len(audio) / samplerate))

    # Initialize the pipeline with the audio classification model
    pipe = pipeline("audio-classification", model="mtg-upf/discogs-maest-30s-pw-129e", trust_remote_code=True)

    # Perform classification
    results = pipe(audio)

    # Output handling
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
