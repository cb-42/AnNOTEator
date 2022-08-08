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

<img src="https://github.com/cb-42/siads_697_capstone_annoteators/blob/main/Flow_diagram.png" 
     alt="Annoteators Flow Diagram" width="740">


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

