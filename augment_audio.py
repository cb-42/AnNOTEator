## This script contains functions to facilitate audio data augmentation for The AnNOTEators Project ##

## Library import ##
from librosa.effects import pitch_shift
from numpy.random import choice

def augment_pitch(audio_clip, sample_rate=22050, n_steps=3, step_var=None, bins_per_octave=24,
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
    if len(audio_clip) == 0:
        return audio_clip # Handle case where empty list (no audio) is supplied
    
    if step_var is not None:
        n_steps += choice(step_var)
    
    return pitch_shift(audio_clip, sr=sample_rate, n_steps=n_steps, bins_per_octave=bins_per_octave, res_type=res_type)
