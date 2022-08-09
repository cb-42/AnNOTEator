# The AnNOTEators Capstone Project
Greetings! This is our Summer 2022 Capstone Project for the Master of Applied Data Science at the University of Michigan School of Information. Our goal is to predict drum notes from audio to create sheet music. The team consists of Christopher Brown, Stanley Hung, and Severus Chang.  

# Getting Started
- tbd

## Software
Our requirements.txt file has a list of the Python library dependencies needed to run our Python scripts and Jupyter notebooks. Due to differing version dependencies in the model training and audio inference portions of the workflow, two environments are recommended.

A Docker image for audio input processing for inference can be acquired from the project [Docker Hub repository](https://hub.docker.com/r/cbrown42/annoteators_project). 

Note that if you wish to use the Python `Spleeter` library for audio data preparation there are additional dependencies, such as ffmpeg, as noted [here](https://pypi.org/project/spleeter/).


# Introduction
- tbd

# How this works?

<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/Flow_diagram.jpg" 
     alt="Annoteators Flow Diagram" width="740">

For a more detailed explanation of each step, please visit our blog post here

# How to start?

There are a few ways to install and use this package

### Interactive Web App
- tbd

### Run from local computer
First download this repo or git clone this repo to your local computer

```bash
#please make sure you already have Github CLI installed
gh repo clone cb-42/siads_697_capstone_annoteators
#navigate to the root directory and install the necessary packages
pip install -r requirements.txt
```

Below is a quick demo code of tranascribing a song to drum sheet music
```python
#import packages
from inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
from inference.transcriber import drum_transcriber

#If you want to use the audio from a Youtube video...
path = get_yt_audio("Youtube link of your choice") 

#Or specify the file path to the audio file in your compauter
path = "the path to the audio file in your compauter"

#Extract drum track from the Audio File / Youtube Audio
drum_track, sample_rate=drum_extraction(path, kernel='demucs') #Recommend to use demucs kernel. For more information about the kernel choise, please check help(drum_extraction).

#Create dataframe for prediction task
df, bpm=drum_to_frame(drum_track, sample_rate) #specifying a few parameters here would help boosting prediction accuracy. For more information about the parameter details, please check help(drum_to_frame).

#Serv to add prediction step here

#The output prediction labels and relevant meta info will be used to construct the sheet music
song_duration = librosa.get_duration(drum_track, sr=sample_rate)
sheet_music = drum_transcriber(prediction_df, song_duration, bpm, sample_rate)

#If you are in the notebook enviornment, you can render the sheet music directly in the notebook.
#To render or export sheet music in pdf format, Musescore3 software(https://musescore.org/en/download) need to be installed beforehand.   
sheet_music.sheet.show('text') #display the MusicXML file in text format
sheet_music.sheet.show() #display the sheet music directly in the notebook

sheet_music.sheet.write() #export the sheet music in MusicXML format
sheet_music.sheet.write(fmt='musicxml.pdf') #export the sheet music in pdf
```

# Pre-trained Model
The pre-trained model is a convolutional neural network (ConvNet) model that trained on the Expanded Groove MIDI Dataset (E-GMD) from Google Magenta project.

## Source data  
This project used The Expanded Groove MIDI Dataset (E-GMD) for model development. E-GMD Dataset is a large dataset of human drum performances, with audio recordings annotated in MIDI. E-GMD contains 444 hours of audio from 43 drum kits and is an order of magnitude larger than similar datasets. It is also the first human-performed drum transcription dataset with annotations of velocity.

The E-GMD dataset was developed by a group of Google Researchers. For more information about the dataset, please visit their site: [The Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd).
## Data Processing
- The E-GMD dataset consist of 

## Data Augmentation
- tbd

## Model Architecture
- tbd

## Evaluation
- tbd

# Additional Resources
- tbd

# Reference
This software uses the following open source packages:

- [Demucs](https://github.com/facebookresearch/demucs)
- [Librosa](https://github.com/librosa/librosa)
- [Mido](https://github.com/mido/mido/)
- [Music21](https://github.com/cuthbertLab/music21)
- [Spleeter](https://github.com/deezer/spleeter)

