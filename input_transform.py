from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import librosa
import pandas as pd
from pytube import YouTube
import warnings

def drum_extraction(path, music_start=None, music_end=None):
    """
    This is a function to transform the input audio file into a ready-dataframe for prediction task  
    :param path (str): the path to the audio file
    :param music_start (int): the start of the music in the file (in seconds). If not set, assume to start at the begining of the track
    :param music_end (int): the end of the music in the file (in seconds). If not set, assume to end at the end of the track

    :return drum_tracck (numpt array): the extracted drum track
    :return sample_rate (int): the sampling rate of the extracted drum track
    """

    
    #default to use 4stems pre-train model from the Spleeter package for audio demixing 
    separator = Separator('spleeter:4stems')

    audio_adapter = AudioAdapter.default()
    #extract sampling rate from the audio file using the librosa package 
    
    if music_start!= None or music_end!=None:
        if isinstance(music_start, type(None)):
            raise ValueError('Please specify the music start time (in seconds) of your file / Youtube link')
        if isinstance(music_end, type(None)):
            raise ValueError('Please specify the music end time (in seconds) of your file / Youtube link')
        y, sr=librosa.load(path, offset=music_start, duration=music_end-music_start, sr=None)
        sample_rate = sr
        waveform, _ = audio_adapter.load(path, offset=music_start, duration=music_end-music_start, sample_rate=sample_rate)
    else:
        y, sr=librosa.load(path, sr=None)
        sample_rate = sr
        waveform, _ = audio_adapter.load(path, sample_rate=sample_rate)
        
    
    prediction = separator.separate(waveform)

    #use librosa onset_detection algorithm to extract drum hit
    drum_track=librosa.to_mono(prediction["drums"].T)
    return drum_track, sample_rate

def drum_to_frame(drum_track, sample_rate, estimated_bpm=None, resolution=8, fixed_clip_length=True, hop_length=1024):

    """
    This is a function to detect and extract onset from a drum track and format the onsets into a df for prediction task 
    :param drum_track (numpy array): the extracted drum track
    :param sample_rate (int): the sampling rate of the drum track
    :param estimated_bpm (int): beat per minute. it is best to provide a estimated bpm to improve the bpm detection accuracy
    :param resolution (int): either 8/16/32. default 8. control the window size of the onset sound clip if "fixed_clip_length" is not set. 8 means the window size equal to the 8th note duration (calculated by the bpm value), etc.
    :param fixed_clip_length (bool): default True. set window_size of the clip to 0.2 seconds as default, override resolution setting if set to True.

    :return df (pd dataframe): the dataframe that contains the information of all onset found in the track
    :return bpm (float): the estimated bpm value
    """

    if fixed_clip_length==False:      
        if estimated_bpm==None:
            warnings.warn('If fixed_clip_length is False, It is strongly reommended to provide an estimated BPM value, even if it is just a proxy.')
            print('-----------------------------')
            print('BPM value not set......BPM will be estimated by the default algorithm, which may not be reliable in some cases.')
            print('Please note that inaccurate BPM could lead to miscalculation of note duration and poor model performancce.')
        print('-----------------------------')
        print(f'resolution = {resolution}. ')
        print(f'{resolution} note duration is set, this means the duration of the sliced audio clip will have the same duration as an {resolution} note in the song')
        print('It is recommended to set the resolution value either 8 or 16, if not familiar with song structure')
        print('-----------------------------')
    
    o_env = librosa.onset.onset_strength(drum_track, sr=44100, hop_length=hop_length)
    onset_frames=librosa.onset.onset_detect(drum_track, onset_envelope=o_env, sr=sample_rate, backtrack=True)
    peak_frames=librosa.onset.onset_detect(drum_track, onset_envelope=o_env, sr=sample_rate)
    onset_times=librosa.frames_to_time(onset_frames, sr=sample_rate, )
    onset_samples = librosa.frames_to_samples(onset_frames*(hop_length/512))
    peak_samples = librosa.frames_to_samples(peak_frames*(hop_length/512))
    
    #calculate note duration for 4,8,16,32 note with respect to the bpm of the song
    if estimated_bpm != None:
        bpm=librosa.beat.tempo(drum_track, sr=sample_rate, start_bpm=estimated_bpm)[0]
    else:
        bpm=librosa.beat.tempo(drum_track, sr=sample_rate)[0]
        
    q_note_duration=60/bpm
    eigth_note_duration=60/bpm/2
    sixteenth_note_duration=60/bpm/4
    thirty_second_note_duration=60/bpm/8
    
    if resolution==4:
        window_size=librosa.time_to_samples(q_note_duration, sr=sample_rate)
    elif resolution==8:
        window_size=librosa.time_to_samples(eigth_note_duration, sr=sample_rate)
    elif resolution==16:
        window_size=librosa.time_to_samples(sixteenth_note_duration, sr=sample_rate)
    elif resolution==32:
        window_size=librosa.time_to_samples(thirty_second_note_duration, sr=sample_rate)
    else:
        raise ValueError('The resolution must be either 4,8,16 or 32') 
    
    if fixed_clip_length==True:
        window_size=librosa.time_to_samples(0.2, sr=sample_rate)
    # create df for prediction task
    df_dict={'audio_clip':[],
        'sample_start':[],
        'sample_end':[],
        'sampling_rate':[]}

    for onset in onset_samples:
        df_dict['audio_clip'].append(drum_track[onset:onset+window_size])
        df_dict['sample_start'].append(onset)
        df_dict['sample_end'].append(onset+window_size)
        df_dict['sampling_rate'].append(sample_rate)

    df=pd.DataFrame.from_dict(df_dict)
    df['peak_sample']=pd.Series(peak_samples)

    #check clip length to align with model requirement
    def resampling(x, target_length):
        org_sr=x['sampling_rate']
        tar_sr_ratio=target_length/len(x['audio_clip'])
        return pd.Series([librosa.resample(x['audio_clip'], orig_sr=org_sr, target_sr=int(org_sr*tar_sr_ratio)), int(org_sr*tar_sr_ratio)])
    df[['audio_clip', 'sampling_rate']]=df.apply(
            lambda x:resampling(x, 8820) if len(x['audio_clip'])!=8820 else pd.Series([x['audio_clip'], x['sampling_rate']]) , axis=1)


    return df, bpm


def get_yt_audio(link):
    yt = YouTube(link)
    stream=yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    path=stream.download()
    return path