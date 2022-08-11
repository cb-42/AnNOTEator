# The AnNOTEators Capstone Project
Greetings! This is our Summer 2022 Capstone Project for the Master of Applied Data Science at the University of Michigan School of Information. Our goal is to predict drum notes from audio to create sheet music. The team consists of Christopher Brown, Stanley Hung, and Severus Chang.  

## Software
Our requirements.txt file has a list of the Python library dependencies needed to run our Python scripts and Jupyter notebooks. Due to differing version dependencies in the model training and audio inference portions of the workflow, two environments are recommended.

A Docker image for audio input processing for inference can be acquired from the project [Docker Hub repository](https://hub.docker.com/r/cbrown42/annoteators_project). 

Note that if you wish to use the Python `Spleeter` library for audio data preparation there are additional dependencies, such as `ffmpeg`, as noted [here](https://pypi.org/project/spleeter/).


# Introduction
Sheet music is a fundamental and important tool for most musicians. It makes individuals much faster and more efficient in preparing to play. Nowadays, obtaining properly written sheet music of a song could be troublesome unless that song is particularly popular and in the worst case a musician need to transcribe it themselves. The AnNOTEators project aims to help with this situation by leveraging neural networks to automatically transcibe each instrument part in a song. Due to the 8 week time limit for this project, the team decided to focus on transcribing drum notes and produce drum sheet music from a given song, rather than handle all instrument layers. You can find more details of the piepline and framework in the [How this works](https://github.com/cb-42/siads_697_capstone_annoteators#how-this-works) section. We may expand the scope of this project to cover more instrument components in the future.

It is important to check out the [Known issues and limitations](https://github.com/cb-42/siads_697_capstone_annoteators#known-issues-and-limitations) sections for more information about factors to be aware of when using this package.

To learn more about the technical details of this project, please visit our [blog post]. **Attach link later**

# How does this work?

<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/Flow_diagram.jpg" 
     alt="Annoteators Flow Diagram" width="740">

For a more detailed explanation of each step, please visit our [blog post]. **Attach link later**

# Getting Started

There are a few ways to install and use the code, models, and environments we've developed.

### Interactive Web App
- tbd

### Docker image

Docker images for the model training and the audio inference environments can be acquired from our [Docker Hub repository](https://hub.docker.com/r/cbrown42/annoteators_project). These come with necessary Python libraries and other software (like MuseScore) pre-installed.

For example, to pull the inference image, use the following command:  

```bash
docker pull cbrown42/annoteators_project:inference-0.02
```


### Command Line Interface

```bash
#please make sure you already have Github CLI installed
gh repo clone cb-42/siads_697_capstone_annoteators

#navigate to the root directory and install the necessary packages
pip install -r requirements.txt

#Below code will transcribe the audio from the youtube link below and output the pdf file in the root directory
#Musescore 3 need to be installed for pdf export
python main.py -i 'https://www.youtube.com/watch?v=hTWKbfoikeg' -o 'pdf'

```

### Notebook Enviornment
First download this repo or git clone this repo to your local computer

```bash
# Please make sure you already have Github CLI installed
gh repo clone cb-42/siads_697_capstone_annoteators
# Navigate to the root directory and install the necessary packages
pip install -r requirements.txt
```

Below is a quick demo code of tranascribing a song to drum sheet music
```python

from inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
from inference.transcriber import drum_transcriber

# If you want to use the audio from a Youtube video...
# It is recommended to use offical MV / high quality audio as the input.
# Avoid using live performance versions which could cause unreliable / strange sheet music results.
path = get_yt_audio("Youtube link of your choice") 

# Or specify the file path to the audio file in your compauter
path = "the path to the audio file in your compauter"

# Extract drum track from the Audio File / Youtube Audio
drum_track, sample_rate = drum_extraction(path, kernel='demucs') 
# We recommend using the demucs kernel. 
# Please check help(drum_extraction) for further details about the kernels

# Create dataframe for prediction task
df, bpm = drum_to_frame(drum_track, sample_rate) 
# To improve prediction accuracy,
# Please check help(drum_to_frame) for instruction on fine tuning parameters 

# Serv to add prediction step here

# The output prediction labels and relevant meta info will be used to construct the sheet music
song_duration = librosa.get_duration(drum_track, sr=sample_rate)
sheet_music = drum_transcriber(prediction_df, song_duration, bpm, sample_rate)

# If you are in the notebook enviornment, you can render the sheet music directly in the notebook.
# To render or export sheet music in pdf format, 
# Musescore3 software (https://musescore.org/en/download) needs to be installed beforehand.   
sheet_music.sheet.show('text') # display the MusicXML file in text format
sheet_music.sheet.show() # display the sheet music directly in the notebook

sheet_music.sheet.write() # export the sheet music in MusicXML format
sheet_music.sheet.write(fmt='musicxml.pdf') # export the sheet music in pdf

```

# Custom training and pre-trained Model (For Data Scientists)
We have uploaded a pre-trained model in the <folder>, which is used in the prediction by default. The pre-trained model is a convolutional neural network (ConvNet) model that trained on the Expanded Groove MIDI Dataset (E-GMD) from the Google Magenta project. We also provided all the nessesary tooling for anyone that wishes to replicate or modify the training pipeline.  

## Source data  
This project used The Expanded Groove MIDI Dataset (E-GMD) for model development. E-GMD Dataset is a large dataset of human drum performances, with audio recordings annotated in MIDI. E-GMD contains 444 hours of audio from 43 drum kits and is an order of magnitude larger than similar datasets. It is also the first human-performed drum transcription dataset with annotations of velocity.

The E-GMD dataset was developed by a group of Google Researchers. For more information about the dataset, please visit their site: [The Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd).
## How were the data processed for model training? 
<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/data_preparation.jpg" 
     alt="Data Processing Diagram" width="740">

- Each drum track record in the dataset consist of 2 files - a MIDI file and a WAV audio file. The MIDI file and WAV file were synced to within 2ms time differences
- The WAV audio was sliced into a series of mini audio clips with the relevant label captured from the MIDI messages. 
- Each audio clip represents the sound of a single drum hit.
- Please refer to the `data_preparation.py` script for more details. We also prepared a notebook to showcase how data preparation elements work and connect together.

## Data Augmentation
- Augmentation is not recommended for model development.

## Model Architecture
- Serv to add

## Evaluation
- Serv to add

# Additional Resources
- tbd

# Known issues and limitations
- The model has a poor performance in predicting multi-hit labels due to the lack of multi-hit labeled data in the training set. This could be fixed by modifying the data preparation algorithm.
- The quantization and time mapping algorithm may not be 100% accurate all the time. This approach is also very sensitive to the 'exact time' of each hit in the track. A slight delay (which always happens in human performed drumplay) sometimes could make the note duration detection go wrong. For example, a triplet note could be detected as a 16th note, due to a very little delay. A hidden markov chain model could be a solution to fix this problem - please visit our blog post for a deeper dive discussion on this.
- There is no standard style in writing drum sheet music. This project implemented a style of our choice, which may not suit everyone's styling preference. To change the notation style, it is necessary to modify the code in the transcriber script. This package uses the `Music21` package for sheet music construction.
- The standalone drum track demixed by `demucs` is not an original drum track. Some audio features could be altered or lost entirely during the demixing process. It is a known issue that the `demucs` processed drum track has a 'much cleaner signal' than the training drum track, which caused the prediction accuracy issue we observed. Please visit our blog post for a deeper dive and discussion about this, as well as the proposed methodology to fix this issue. 

# Reference
This software uses the following open source packages:

- [Demucs](https://github.com/facebookresearch/demucs)
- [Librosa](https://github.com/librosa/librosa)
- [Mido](https://github.com/mido/mido/)
- [Music21](https://github.com/cuthbertLab/music21)
- [Pedalboard](https://github.com/spotify/pedalboard)
- [Spleeter](https://github.com/deezer/spleeter)
