## This script contains functions to facilitate audio data augmentation for The AnNOTEators Project ##

import librosa.display
from librosa.effects import pitch_shift
import matplotlib.pyplot as plt
from numpy import random
from pedalboard import LowpassFilter, Pedalboard, Reverb


def add_pedalboard_effects(audio_clip, sample_rate=44100, pb=None, room_size=0.6, cutoff_freq=1000):
    """
    Add pedalboard effects to an audio file.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create().
        sample_rate: This number specifies the audio sampling rate.
        pb: pedalboard.Pedalboard initialized with reverb, lowpass filter, and/or other effects. If no Pedalboard object
            is passed, then one will be created using Reverb and LowpassFilter effects.
        room_size: Float to be passed to Reverb effect used to simulate the room space.
        cutoff_freq: Value to use as the lowpass filter frequency cutoff.
        
    Returns:
        An augmented numpy.ndarray, with Pedalboard effects applied.
        
    Example usage:
        rev_clip = add_pedalboard_effects(clip, sample_rate, room_size=0.7)
        audio_df['pb_aug_audio'] = audio_df.audio_wav.progress_apply(lambda x: add_pedalboard_effects(x, sample_rate))
    """
    if pb is None:
        pb = Pedalboard([Reverb(room_size), LowpassFilter(cutoff_frequency_hz=cutoff_freq)])
    
    return pb(audio_clip, sample_rate)


def add_white_noise(audio_clip, noise_ratio=.1, random_state=None):
    """
    Add white noise to an audio signal, scaling by a noise ratio.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create().
        noise_ratio: Floating point to use for scaling the white noise. Higher values will increasingly 
            overwhelm the original signal.
        random_state: Integer seed to use for reproducibility.
            
    Returns:
        An augmented numpy.ndarray, with white noise added according to the noise ratio.
        
    Example usage:
        wn_clip = add_white_noise(clip, .05)
        audio_df['wn_audio'] = df.audio_wav.progress_apply(lambda x: add_white_noise(x, noise_ratio=0.5))
    """
    if random_state is not None:
        random.seed(seed=random_state)
    
    wn = random.normal(loc=0, scale=audio_clip.std(), size=audio_clip.shape[0])
    return audio_clip + wn * noise_ratio


def augment_pitch(audio_clip, sample_rate=44100, n_steps=3, step_var=None, bins_per_octave=24,
                  res_type='kaiser_best'):
    """
    Augment the pitch of an audio file.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create().
        sample_rate: This number specifies the audio sampling rate.
        n_steps: The number of steps to shift the pitch of the audio clip.
        step_var: Optionally supply a range of integers to allow variation in the number of steps taken.
        bins_per_octave: This is the number of steps per octave.
        res_type: (str) The resampling strategy to use. By default, the highest quality option is used.
    
    Returns:
        An augmented numpy.ndarray, pitch shifted according to specified parameters.
        
    Example usage:
        aug_clip = augment_pitch(clip, n_steps=2, step_var=range(-1, 2, 1))
        audio_df['augmented_pitch'] = audio_df.audio_wav.progress_apply(lambda x: augment_pitch(x, n_steps=2, step_var=range(-1, 2, 1)))
    """
    if step_var is not None:
        n_steps += random.choice(step_var)
    
    return pitch_shift(audio_clip, sr=sample_rate, n_steps=n_steps, bins_per_octave=bins_per_octave, res_type=res_type)


def compare_waveforms(df, i, signal_cols, signal_labs=None, alpha=0.5, figsize=(16,12)):
    """
    Visually compare various augmentations of the same signal (or other signals after resampling) using
    librosa.display.waveform.
    
    Parameters:
        df: Dataframe to use to retrieve signal columns.
        i: Index of audio clip to plot.
        signal_cols: List of column names to retrieve signals from. The order should match labels in signal_list. These are assumed
            to be in numpy array format, resulting from librosa processing. Current code expects any white noise signal listed first,
            so other signals can be viewed more readily.
        signal_labs: A list of strings, which are brief descriptors to use as labels for each signal.
        alpha: Alpha setting to use for white noise signal, which can improve visibility of other signals. This can supplied
            as a list, with a separate value for each signal. Otherwise, the same alpha value will be used for all signals.
        figsize: Figure size tuple to pass to plt.figure.
        
    Example usage:
        compare_waveforms(df=sub_df, i=0, signal_cols=['wn_audio', 'audio_wav', 'augmented_pitch'],
                signal_labs=['white noise', 'original', 'pitch shift'], alpha=[0.3, 0.7, 0.7])
        
    """
    if signal_labs is None:
        signal_labs = signal_cols
    elif (len(signal_labs) < len(signal_cols)) | (type(signal_labs) != list):
        print('Not enough labels were provided in list form. Using column names as labels.')
        signal_labs = signal_cols
        
    if type(alpha) == float:
        alpha = [alpha for s in signal_cols]
    elif len(alpha) < len(signal_cols):
        print('Alpha list contained insufficient values, using first value for all signals.')
        alpha = [alpha[0] for s in signal_cols]
    
    plt.figure(figsize=figsize)
    
    for col, lab, alp in list(zip(signal_cols, signal_labs, alpha)):
        librosa.display.waveplot(df.loc[i, col], label=lab, alpha=alp) 
    
    plt.legend(signal_labs)    
    plt.title('Comparison of audio clips for element: {}, label: {}'.format(i, df.loc[i, 'label']))
