# The AnNOTEators Capstone Project
Greetings! This is our Summer 2022 Capstone Project for the Master of Applied Data Science at the University of Michigan School of Information. Our goal is to predict drum notes from audio to create sheet music. The team consists of Christopher Brown, Stanley Hung, and Severus Chang.  

## Software
Our requirements.txt file has a list of the Python library dependencies needed to run our Python scripts and Jupyter notebooks. Due to differing version dependencies in the model training and audio inference portions of the workflow, two environments are recommended.

A Docker image for audio input processing for inference can be acquired from the project [Docker Hub repository](https://hub.docker.com/r/cbrown42/annoteators_project). 

Note that if you wish to use the Python `Spleeter` library for audio data preparation there are additional dependencies, such as ffmpeg, as noted [here](https://pypi.org/project/spleeter/).


# Introduction
Sheet music is a fundamental and important tool for most musicians. It made you much faster and more efficient in preparing the play. Nowadays, obtaining properly written sheet music of a song could be troublesome unless that song is particularly popular, the worst case is that the musician need to transcribe it by themeselves. The AnNOTEators project aims to help with this situation by leveraging neural network to automatically transcibe each instrument part in a song. Due to the time limit (8 weeks) of this project, the team decided to start with transcrbing drum notes and produce drum sheet music from a given song. You can find more details of the piepline and framework in the [How this works](https://github.com/cb-42/siads_697_capstone_annoteators#how-this-works) section. We may expand the scope of this project to cover more instrument parts in the future.

It is important to check out the [Known issues and limitations](https://github.com/cb-42/siads_697_capstone_annoteators#known-issues-and-limitations) sections for more information about the things you need to be aware of when using this package.

If you are interested to learn mroe about the technical details of this project, please visit our blog post. Attach link later 

# How this works?

<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/Flow_diagram.jpg" 
     alt="Annoteators Flow Diagram" width="740">

For a more detailed explanation of each step, please visit our blog post here

# Getting Started

There are a few ways to install and use this package

### Interactive Web App
- tbd

### Docker image
- Chris to add

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
#please make sure you already have Github CLI installed
gh repo clone cb-42/siads_697_capstone_annoteators
#navigate to the root directory and install the necessary packages
pip install -r requirements.txt
```

Below is a quick demo code of tranascribing a song to drum sheet music
```python

from inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
from inference.transcriber import drum_transcriber

# If you want to use the audio from a Youtube video...
# It is recommended to use offical MV / high quality audio as an input.
# Avoid using live performance version which could cause unreliable / strange sheet music result.
path = get_yt_audio("Youtube link of your choice") 

# Or specify the file path to the audio file in your compauter
path = "the path to the audio file in your compauter"

# Extract drum track from the Audio File / Youtube Audio
drum_track, sample_rate=drum_extraction(path, kernel='demucs') 
# Recommend to use demucs kernel. 
# Please check help(drum_extraction) for further details about the kernels

# Create dataframe for prediction task
df, bpm=drum_to_frame(drum_track, sample_rate) 
# To improve the prediction accuracy,
# Please check help(drum_to_frame) for instruction on fine tuning parameters 

# Serv to add prediction step here

# The output prediction labels and relevant meta info will be used to construct the sheet music
song_duration = librosa.get_duration(drum_track, sr=sample_rate)
sheet_music = drum_transcriber(prediction_df, song_duration, bpm, sample_rate)

# If you are in the notebook enviornment, you can render the sheet music directly in the notebook.
# To render or export sheet music in pdf format, 
# Musescore3 software(https://musescore.org/en/download) need to be installed beforehand.   
sheet_music.sheet.show('text') # display the MusicXML file in text format
sheet_music.sheet.show() # display the sheet music directly in the notebook

sheet_music.sheet.write() # export the sheet music in MusicXML format
sheet_music.sheet.write(fmt='musicxml.pdf') # export the sheet music in pdf

```

# Custom training and pre-trained Model (For Data Scientist)
We have ploaded a pre-trained model in the <folder>, which is used in the prediction by default. The pre-trained model is a convolutional neural network (ConvNet) model that trained on the Expanded Groove MIDI Dataset (E-GMD) from Google Magenta project. We also provided all the nessesary tooling if you wish to replicate / modify the training pipeline.  

## Source data  
This project used The Expanded Groove MIDI Dataset (E-GMD) for model development. E-GMD Dataset is a large dataset of human drum performances, with audio recordings annotated in MIDI. E-GMD contains 444 hours of audio from 43 drum kits and is an order of magnitude larger than similar datasets. It is also the first human-performed drum transcription dataset with annotations of velocity.

The E-GMD dataset was developed by a group of Google Researchers. For more information about the dataset, please visit their site: [The Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd).
## How the data were processed for model training? 
<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/data_preparation.jpg" 
     alt="Data Processing Diagram" width="740">

- Each drum track record in the dataset consist of 2 files - a MIDI file and a WAV audio file. The MIDI file and WAV file were synced to within 2ms time differences
- The WAV audio was slicied into a series of mini audio clips with the relevant label captured from the MIDI messages. 
- Each audio clip represent a sound of a single drum hit
- Please refer to the data_preparation script for more details. We also prepared a notebook to showcase how things work and connect together.

## Data Augmentation
- Augmentation is not recommended for model development.

## Model Architecture
- Serv to add

## Evaluation
- Serv to add

# Additional Resources
- tbd

# Known issues and limitations
- The model has a poor performance in predicting multi-hit label due to the lack of multi-hit labeled data in the training set. This could be fixed by modifying the data preparation algorithm.
- The quantization and time mapping algorithm may not be 100% accurate all the time. This approach is also very sensitive to the 'exact time' of each hit in the track. A slight delay (which always happend in human performed drumplay) sometimes could make the note duration detection go wrong. For example, a triplet note could be detected as an 16th note, due to a very little delay. Hidden markov chain model could be a solution to fix this problem, please visit our blog post for a deeper dive discussion on this 
- There is no standard style in writing drum sheet music. This project implemented a style of our choice, which may not suit everyone styling preference. To change the notation style, you will need to modify the code in the transcriber script. This package use the `Music21` package for sheet music construction.
- The standalone drum track demixed by `demucs` is not an original drum track. Some audio features could have changed or losed during the process. It is a known issue that the `demucs` processed drum track has a 'much cleaner signal' than the training drum track, which caused the prediction accuracy issue. Please visit our blog post for a deeper dive discussion on this, as well as the proposed methodology to fix this. 
# Reference
This software uses the following open source packages:

- [Demucs](https://github.com/facebookresearch/demucs)
- [Librosa](https://github.com/librosa/librosa)
- [Mido](https://github.com/mido/mido/)
- [Music21](https://github.com/cuthbertLab/music21)
- [Spleeter](https://github.com/deezer/spleeter)

