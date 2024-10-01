# EDM genre detector

* Short Answer - This is a CLI wrapper of [`discogs-maest-30s-pw-129e`](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e).
* Longer Answer - This tool performs audio classification using a pre-trained model ([`discogs-maest-30s-pw-129e`](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e)) from the Hugging Face `transformers` library. It is designed to handle WAV audio files, process them appropriately, and output the classification results either to the console or to a specified JSON file.

## Features

- **Audio Processing**: Converts stereo to mono and resamples to 16kHz if necessary.
- **Flexible Output**: Outputs classification results to standard output or a JSON file.
- **Command-Line Interface**: Easy to use with command-line arguments for specifying input and output.

## Prerequisites

Before running this tool, ensure you have the following installed:
- `numpy`
- `soundfile`
- `scipy`
- `transformers`

You can install the required libraries using the following command:

```bash
pip install numpy soundfile scipy transformers
```

## Installation

Clone this repository or download the source code to your local machine. No additional installation steps are required.

## Usage

To use the tool, navigate to the directory containing the script and run the following command:

```bash
python audio_classification.py --input path/to/your/file.wav
```

### Arguments

- `--input` (required): Path to the input WAV file.
- `--output` (optional): Path where the output JSON file will be saved.

### Examples

1. **Classify an Audio File and Print Results to Standard Output**
   ```bash
   python audio_classification.py --input /path/to/audio.wav
   ```

2. **Classify an Audio File and Save Results to a JSON File**
   ```bash
   python audio_classification.py --input /path/to/audio.wav --output /path/to/output.json
   ```

## Output

The output will be a JSON formatted string either displayed on the console or saved to a file, detailing the classification results provided by the model.

## Contributing

Contributions are welcome. If you have improvements or bug fixes, please open a pull request with your changes.



