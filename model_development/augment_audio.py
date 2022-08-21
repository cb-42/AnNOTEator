## This script contains functions to facilitate audio data augmentation for The AnNOTEators Project ##

import librosa.display
from librosa.effects import pitch_shift
from math import sqrt
import matplotlib.pyplot as plt
from numpy import mean, random
from pedalboard import LowpassFilter, Pedalboard, Reverb


def add_lowpass_filter(audio_clip, sample_rate=44100, cutoff_freq=1000):
    """
    Add pedalboard.LowpassFilter to an audio file.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create_audio_set().
        sample_rate: This number specifies the audio sampling rate.
        cutoff_freq: Value to use as the lowpass filter frequency cutoff.
        
    Returns:
        An augmented numpy.ndarray, with LowpassFilter applied.
        
    Example usage:
        lowpass_clip = add_lowpass_filter(clip, sample_rate, cutoff_freq=1000)
        audio_df['lowpass_filter'] = audio_df.audio_wav.progress_apply(lambda x: add_lowpass_filter(x, sample_rate))
    """
    pb = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_freq)])
    
    return pb(audio_clip, sample_rate)


def add_pedalboard_effects(audio_clip, sample_rate=44100, pb=None, room_size=0.6, cutoff_freq=1000):
    """
    Add pedalboard effects to an audio file.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create_audio_set().
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


def add_white_noise(audio_clip, snr=10, random_state=None):
    """
    Add white noise to an audio signal, scaling by a signal to noise ratio.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create_audio_set().
        snr: Target value to use for signal to noise ratio.
        random_state: Integer seed to use for reproducibility.
            
    Returns:
        An augmented numpy.ndarray, with white noise added according to the noise ratio.
        
    Example usage:
        wn_clip = add_white_noise(clip, random_state=random_state)
        audio_df['wn_audio'] = df.audio_wav.progress_apply(lambda x: add_white_noise(x, snr=20))
    """
    if isinstance(random_state, int):
        random.seed(seed=random_state)
    
    audio_clip_rms = sqrt(mean(audio_clip**2))
    noise_rms = sqrt(audio_clip_rms**2/(10**(snr/10)))
    white_noise = random.normal(loc=0, scale=noise_rms, size=audio_clip.shape[0])
    
    return audio_clip + white_noise


def apply_augmentations(df, audio_col='audio_wav', aug_col_names=None, **aug_param_dict):
    """
    Helper function for applying arbitrary number of augmentations to a dataframe containing audio.
    
    Parameters:
        df: Audio dataframe containing the audio_col specified and in which augmentations will be stored.
        audio_col: String corresponding to the name of the column to use for audio data augmentation,
        in numpy array format.
        aug_col_names: Names to use for augmented columns. Defaults to using the augmentation functions
            as column names.
        aug_param_dict: Dictionary of function names and associated parameters.
    
    Returns:
        A dataframe with new columns containing augmented numpy arrays.
        
    Example usage:
        aug_params = {
            'add_white_noise': {'snr':20, 'random_state': random_state},
            'augment_pitch': {'n_steps':2, 'step_var':range(-1, 2, 1)},
            'add_pedalboard_effects': {}
            }
        apply_augmentations(audio_df, col_names=aug_cols, **aug_params)
    """
    aug_df = df.copy(deep=True)
    
    for func, params in aug_param_dict.items():
        print('Applying {}'.format(func))
        aug_df[func] = aug_df[audio_col].progress_apply(lambda x: eval(func)(x, **params))
        
    if aug_col_names is not None:
        col_dict = {}
        for fun, col in zip(aug_param_dict.keys(), aug_col_names):
            col_dict[fun] = col
        aug_df.rename(columns=col_dict, inplace=True)
        
    return aug_df


def augment_pitch(audio_clip, sample_rate=44100, n_steps=3, step_var=None, bins_per_octave=12,
                  res_type='kaiser_best'):
    """
    Augment the pitch of an audio file.
    
    Parameters:
        audio_clip: The audio clip is presumed to be a single wav file resulting from data_preparation.create_audio_set().
        sample_rate: This number specifies the audio sampling rate.
        n_steps: The number of steps to shift the pitch of the audio clip.
        step_var: Optionally supply a range of integers to allow variation in the number of steps taken.
        bins_per_octave: This is the number of steps per octave. By using the 12 bin default, a step is equivalent to a semitone.
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


def augment_spectrogram_spans(spec, spans=3, span_ranges=[[1,4], [1,6]], span_variation=1,
                              ind_lists=None, sig_val=None, overwrite=False):
    """
    Set spectrogram (or other 2-d numpy array) span(s) to background or another signal.
    
    Parameters:
        spec: Numpy array; presumed to be a spectrogram or other signal approprtiate for dropout augmentation.
        spans: Integer corresponding to the number of dropout spans to construct.
        span_ranges: List of lists corresponding to integers for x and y dimensions used in spans.
        span_variation: Integer corresponding to variation to introduce into span lengths.
        ind_lists: List of lists corresponding to array indices to augment; By default, a list will be
            created.
        sig_val: value to set sub arrays to; minimum (background) value is used by default.
        overwrite: Boolean signifying whether to overwrite the input array.
    
    Returns:
        An augmented spectrogram (or other numpy array) with dropout spans applied.
        
    Example usage (note that input should be copied to avoid overwriting the original signal, if desired):
        audio_df['mel_spec_dropout'] = audio_df.mel_spec.progress_apply(lambda x: augment_spectrogram_spans(x.copy)))
    
    """
    if overwrite==False:
        spec = spec.copy()
    
    if not sig_val:
        sig_val = spec.min() # use for setting to background
        
    if not isinstance(spans, int):
        print('Value supplied for spans supplied was not an integer. Setting spans to 1.')
        spans = 1
    
    if not ind_lists:
        ind_lists = []
        dims = spec.shape
        for i in range(0, spans):
            x_inds = get_span_indices(dims[1], min_span=span_ranges[0][0], max_span=span_ranges[0][1],
                                 span_variation=span_variation)
            y_inds = get_span_indices(dims[0], min_span=span_ranges[1][0], max_span=span_ranges[1][1],
                                 span_variation=span_variation)
            ind_lists.append([x_inds, y_inds])
    
    for ind_list in ind_lists: # y then x based on current naming
        spec[ind_list[1][0]:ind_list[1][1]+1:, ind_list[0][0]:ind_list[0][1]+1] = sig_val
    
    return spec


def compare_waveforms(df, i, signal_cols, signal_labs=None, sample_rate=44100, max_pts=None, alpha=0.5, fontsizes=[24, 18, 20], figsize=(16, 12), leg_loc='best', title=''):
    """
    Visually compare the effect of various augmentations on the same signal's amplitude envelope (or other signals 
        after resampling) using librosa.display.waveplot.
    
    Parameters:
        df: Dataframe to use to retrieve signal columns.
        i: Index of audio clip to plot.
        signal_cols: List of column names to retrieve signals from. The order should match labels in signal_list. These are assumed
            to be in numpy array format, resulting from librosa processing. Current code expects any white noise signal listed first,
            so other signals can be viewed more readily.
        signal_labs: A list of strings, which are brief descriptors to use as labels for each signal.
        sample_rate: Sampling rate to be passed to librosa.display.waveplot. Note that we are expecting to use
            44100 as a default, rather than the 22050 default set by waveplot.
        max_pts: Positive integer to pass to waveplot for limiting points to be drawn, which can result in downsampling. For our purposes in working with short clips, we set the default to None to avoid downsampling.
        alpha: Alpha setting to use for white noise signal, which can improve visibility of other signals. This can supplied
            as a list, with a separate value for each signal. Otherwise, the same alpha value will be used for all signals.
        fontsizes: List of font sizes to use for plot title and/or other labels.
        figsize: Figure size tuple to pass to plt.figure, corresponding to titles, axis labels, and legend labels.
        leg_loc: String to pass to plt.legend to control legend positioning.
        title: String to optionally specify the plot title.
        
    Example usage:
        compare_waveforms(df=sub_df, i=0, signal_cols=['wn_audio', 'audio_wav', 'augmented_pitch'],
                signal_labs=['white noise', 'original', 'pitch shift'], alpha=[0.3, 0.7, 0.7], leg_loc='upper left')
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
        librosa.display.waveplot(df.loc[i, col], sr=sample_rate, max_points=max_pts, label=lab, alpha=alp) 
    
    if title == '':
        plt.title('Comparison of audio clips for element: {}, label: {}'.format(i, df.loc[i, 'label']), fontsize=fontsizes[0])
    else:
        plt.title(title, fontsize=fontsizes[0])
    plt.gca().xaxis.label.set_size(fontsizes[1])
    plt.legend(signal_labs, loc=leg_loc, fontsize=fontsizes[2])    

    
def get_span_indices(dim, min_span=5, max_span=None, span_variation=0):
    """
    Helper function to find array indices for spectrogram augmentation.
    
    dim: Integer representing the dimension of an array.
    min_span: Integer for minimum span length; may be less than min_span if lower bound is near 0.
    max_span: Integer for maximum span length, defaults to None. Uses lower bound to set max span.
    span_variation: Integer corresponding to variation to introduce into span lengths.
    
    Returns:
        Span indices
    
    Example usage: 
        get_span_indices(dim=spec.shape[0], min_span=1, max_span=5, span_variation=1)
    """
    
    # Add variation to min/max span lengths
    new_spans = []
    for span_length in (min_span, max_span):
        span_length += random.randint(low=-span_variation, high=span_variation + 1)
        if span_length <= 0:
            span_length = 1
        new_spans.append(span_length)
    min_span, max_span = new_spans
    
    # Create spans
    inds = sorted(random.randint(low=0, high=dim, size=2)) # upper bound not inclusive
    if inds[0] == inds[1]:
        if inds[0] == 0: # account for larger span or add to high end when result would be < 0?
            inds[1] += min(min_span, ((dim-1) - inds[1]))
        else:
            inds[0] -= min(min_span, inds[0])
    else:
        diff = abs(inds[0] - inds[1])
        if diff < min_span: # consider whether to optionally add to end rather than subtract from start
            inds[0] -= min((min_span - diff), inds[0])
        if isinstance(max_span, int):
            if diff > max_span: # consider whether to optionally add to end rather than subtract from start
                inds[1] = (inds[0] + max_span)

    return inds
