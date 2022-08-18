from .augment_audio import add_pedalboard_effects, add_white_noise, apply_augmentations, augment_pitch
import mido
from mido import MidiFile, Message, MidiTrack, MetaMessage
import librosa
import librosa.display
import pandas as pd
import os
from tqdm.notebook import tqdm
import numpy as np
import itertools
import math
import sys


class data_preparation(): 

    tqdm.pandas()

    """
    This is a class for creating training dataset for model training.
    
    :attr directory_path: the directory path specified when initiaed the class
    :attr dataset_type: the type of dataset used 
    :attr dataset: the resampled dataset in pandas dataframe format
    :attr midi_wav_map: the mapping of each midi file and wav file pair
    :attr notes_collection: the training dataset that ready to use in downstrem tasks
    
    :param directory_path (string): The path to the root directory of the dataset. This class assume the use of GMD / E-GMD dataset
    :param dataset (string): either "gmd" or "egmd" is acceptable for now
    :param sample_ratio (float): the fraction of dataset want to be used in creating the training/test/eval dataset
    :param diff_threshold (float): filter out the midi/audio pair that > the duration difference threshold value 

    :raise NameError: the use of other dataset is not supported
    """
    
    def __init__(self, directory_path, dataset, sample_ratio=1, diff_threshold=1):
        if dataset in ['gmd', 'egmd']:
            self.directory_path=directory_path
            self.dataset_type=dataset
            self.batch_tracking=0
            csv_path=[f for f in os.listdir(directory_path) if '.csv' in f][0]
            self.dataset=pd.read_csv(os.path.join(directory_path, csv_path)).dropna().sample(frac=sample_ratio).reset_index()
            df=self.dataset[['index', 'midi_filename', 'audio_filename', 'duration']].copy()
            df.columns=['track_id', 'midi_filename', 'audio_filename', 'duration']
            print(f'Filtering out the midi/audio pair that has a duration difference > {diff_threshold} second')
            df['wav_length']=df['audio_filename'].progress_apply(lambda x:self.get_length(x))
            df['diff']=np.abs(df['duration']-df['wav_length'])
            df=df[df['diff'].le(diff_threshold)]
            self.midi_wav_map=df.copy()
            self.notes_collection=pd.DataFrame()
            # the midi note mapping is copied from the Google project page. note 39,54,56 are new in 
            # EGMD dataset and Google never assigned it to a code. From initial listening test, these
            # are electronic pad / cowbell sound, temporaily assigned it to CB, CowBell for now
            # self.midi_note_map={36:'KD', 38:'SD', 40:'SD', 37:'SD_xstick', 48:'HT', 50:'HT',
            #                    45:'MT', 47:'MT', 43:'FT' ,58:'FT', 46:'HH_open', 
            #                    26:'HH_open', 42:'HH_close', 22:'HH_close', 44:'HH_close',
            #                    49:'CC', 57:'CC', 55:'CC', 52:'CC', 51:'RC',
            #                    59:'RC', 53:'RC', 39:'CB', 54:'CB', 56:'CB'}

            self.midi_note_map={36:'KD', 38:'SD', 40:'SD', 37:'SD', 48:'TT', 50:'TT',
                               45:'TT', 47:'TT', 43:'TT' ,58:'TT', 46:'HH', 
                               26:'HH', 42:'HH', 22:'HH', 44:'HH',
                               49:'CC', 57:'CC', 55:'CC', 52:'CC', 51:'RC',
                               59:'RC', 53:'RC', 39:'CB', 54:'CB', 56:'CB'}

            id_len=self.midi_wav_map[['track_id','wav_length']]
            id_len.set_index('track_id', inplace=True)
            self.id_len_dict=id_len.to_dict()['wav_length']
            
        else:
            raise NameError('dataset not supported')
    def get_length(self,x):
        wav, sr=librosa.load(os.path.join(self.directory_path, x), sr=None, mono=True)
        return librosa.get_duration(wav, sr)

    def notes_extraction(self, midi_file):
        """
        A function to extract notes from the miditrack. A miditrack is composed by a series of MidiMessage.
        Each MidiMessage contains information like channel, note, velocity, time etc.
        This function extract each note presented in the midi track and the associated start and stop playing time of each note
        
        :param midi_file (str):     The midi file path
        :return (list):             a list of 3-element lists, each 3-element list consist of [note label, associated start time in the track, associated end time in the track]  
        """
        
        # the time value stored in the MidiMessage is DELTATIME in ticks unit instead of exact time in second unit.
        # The delta time refer to the time difference between the MidiMessage and the previous MidiMessage
        
        #The midi note key map can be found in here:
        # https://rolandus.zendesk.com/hc/en-us/articles/360005173411-TD-17-Default-Factory-MIDI-Note-Map
        
        self.time_log=0
        notes_collection=[]
        temp_dict={}
        
        self.midi_track=MidiFile(os.path.join(self.directory_path, midi_file))
        
        for msg in self.midi_track.tracks[-1]:

            try:
                self.time_log=self.time_log+msg.time

            except:
                continue

            if msg.type=='note_on' and msg.velocity>0:
                start=self.time_log
                if msg.note in temp_dict.keys():
                    try:
                        temp_dict[msg.note].append(start)
                    except:
                        start_list=[start]
                        start_list.append(temp_dict[msg.note])
                        temp_dict[msg.note]=start_list
                else:
                    temp_dict[msg.note]=start

            elif (msg.type=='note_on' and msg.velocity==0) or msg.type=='note_off':
                end=self.time_log
                if type(temp_dict[msg.note])==list:
                    notes_collection.append([msg.note, math.floor(temp_dict[msg.note][0]*100)/100, math.ceil(end*100)/100])
                    del temp_dict[msg.note][0]
                    if len(temp_dict[msg.note])==0:
                        del temp_dict[msg.note]
                else:
                    notes_collection.append([msg.note, math.floor(temp_dict[msg.note]*100)/100, math.ceil(end*100)/100])
                    del temp_dict[msg.note]

            else:
                continue
        return [[self.midi_note_map[n[0]], n[1], n[2]] for n in notes_collection]
    
    def time_meta_extraction(self):
        """
        A helper function to extract ticks_per_beat and tempo information from the meta messages in the midi file. 
        These information are required to convert midi time into seconds.
        """
        ticks_per_beat=self.midi_track.ticks_per_beat
        for msg in self.midi_track.tracks[0]:
            if msg.type=='set_tempo':
                tempo=msg.tempo
                break
            else:
                pass
        return (ticks_per_beat, tempo)
        
        
    def merge_note_label(self, track_id, notes_collection):
        """
        Merge the notes if they share the same start time, which means these notes were start playing at the same time.
        
        :param track_id (int):          the unique id of the midi_track
        :param notes_collection (list): the list of 3-element lists, each 3-element list consist of
                                        [note label, associated start time in the track, associated end time in the track]

        :return: Pandas DataFrame
        
        """

        merged_note_collection=[]

        key_func = lambda x: x[1]

        for key, group in itertools.groupby(notes_collection, key_func):
            group_=list(group)
            if len(group_)>1:
                merged_note=[x[0] for x in group_]
                start=min([x[1] for x in group_])
                end=max([x[2] for x in group_])
                merged_note_collection.append([merged_note, start, end])
            else:
                merged_note_collection.append(group_[0])

        output_df=pd.DataFrame(merged_note_collection)
        output_df.columns=['label', 'start', 'end']
        output_df['track_id']=track_id
        return output_df

    def ticks_to_second(self, notes_collection):
        """
        A helper function to convert midi time value to seconds. The time value stored in the MidiMessages is in "ticks" unit, this need to be converted
        to "second" unit for audio slicing tasks
        
        :param notes_collection (list):     the list of 3-element lists, each 3-element list consist of
                                            [note label, associated start time in the track, associated end time in the track]
    
        :return: a list of 3-element lists with start and end time rounded to 2 decimal places in seconds
        """
        if type(self.time_log)==float:
            return [[note[0],
                     round(note[1],2),
                     round(note[2],2) ] for note in notes_collection]
        else:
            ticks_per_beat, tempo=self.time_meta_extraction()

            return [[note[0],
                     round(mido.tick2second(note[1],ticks_per_beat,tempo),2),
                     round(mido.tick2second(note[2],ticks_per_beat,tempo),2) ] for note in notes_collection]
    

        
    def create_audio_set(self, pad_before=0.02, pad_after=0.02, fix_length=None, batching=False, dir_path=''):
        """
        main function to create training/test/eval dataset from dataset
        
        :param  pad_before (float):     padding (in seconds) add to the begining of the sliced audio. default 0.02 seconds
        :param  pad_after (float):      padding (in seconds) add to the end of the sliced audio. default 0.02 seconds
                                        The padding actually increas the window length when doing the slicing instead of adding white space before and after.
        :param  fix_length (float):     in seconds, setting this length  will force the sound clip to have exact same length in seconds. suggest value is 0.1~0.2
        :param  batching (bool):        apply batching to avoid memory issues. Suggest to turn this on if processing the full dataset. 
                                        By default, it will divide the dataset into 50 batches and perform train test split automatically. .pkl file will be saved at specified location
        :param  dir_path (str):         The path to the directory to store .pkl files

        :return None
        """
        if batching==True and dir_path==None:
            raise TypeError('please specify directory path for saving pickle files')

        self.batch_tracking=0
        tqdm.pandas()
        df_list=[]
              
        def audio_slicing(x, wav, sr, pad_before, pad_after, window_size=None):
            max_len=len(wav)
            padding_b=librosa.time_to_samples(pad_before, sr=sr)
            padding_a=librosa.time_to_samples(pad_after, sr=sr)
            start=max(librosa.time_to_samples(x['start'], sr=sr)-padding_b, 0)
            if window_size:
                window_size_samples=librosa.time_to_samples(window_size, sr=sr)
                if (start+window_size_samples) > max_len:
                    sliced_wav= wav[start:max_len]
                    return np.pad(sliced_wav, pad_width=(0,start+window_size_samples-max_len), mode='constant', constant_values=0)
                return wav[start:start+window_size_samples]
            else:
                end=min(librosa.time_to_samples(x['end'], sr=sr)+padding_a, len(wav))
            return wav[start:end]

        def resampling(x, target_length):
            org_sr=x['sampling_rate']
            tar_sr_ratio=target_length/len(x['audio_wav'])
            return pd.Series([librosa.resample(x['audio_wav'], orig_sr=org_sr, target_sr=int(org_sr*tar_sr_ratio)), int(org_sr*tar_sr_ratio)])

        def check_length(x, target_length):
            if len(x)>target_length:
                return x[:target_length]
            elif len(x)<target_length:
                return np.pad(x, (0, target_length-len(x)), 'constant')

        def create_df(df_list):
            self.notes_collection=pd.concat(df_list, ignore_index=True)
            self.notes_collection=self.notes_collection[self.notes_collection['audio_wav'].apply(lambda x:len(x))!=0]
            if fix_length!=None:
                pass
            else:
                print('Resampling Audio Data to align data shape')
                target_length=self.notes_collection['audio_wav'].apply(lambda x:len(x)).mode()[0]
                df=self.notes_collection.copy()
                df[['audio_wav_resample', 'resample_sr']]=df.progress_apply(
                    lambda x:resampling(x, target_length) if len(x['audio_wav'])!=target_length else pd.Series([x['audio_wav'], x['sampling_rate']]) , axis=1)
                df['audio_wav_resample']=df.audio_wav_resample.progress_apply(lambda x:check_length(x, target_length) if len(x)!=target_length else x)
                self.notes_collection=df.copy()


            problematic_tracks=[]
            for r in tqdm(self.notes_collection.iterrows(), total=self.notes_collection.shape[0]):
                if r[1].start>self.id_len_dict[r[1].track_id]:
                    problematic_tracks.append(r[1].track_id)
            
            self.notes_collection=self.notes_collection[~self.notes_collection.track_id.isin(problematic_tracks)]



        print('Generating Dataset')
        train=0.6
        val=0.2
        test=0.2

        for key, row in tqdm(self.midi_wav_map.iterrows(), total=self.midi_wav_map.shape[0]):
            if row['midi_filename']=='drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
                continue
            notes_collection=self.notes_extraction(row['midi_filename'])
            converted_notes_collection=self.ticks_to_second(notes_collection)
            track_notes=self.merge_note_label(row['track_id'], converted_notes_collection)
            
        
            wav, sr = librosa.load(os.path.join(self.directory_path, row['audio_filename']), sr=None, mono=True)
            if fix_length!=None:
                track_notes['audio_wav']=track_notes.apply(lambda x:audio_slicing(x, wav, sr, pad_before, pad_after, window_size=fix_length), axis=1)
            else:
                track_notes['audio_wav']=track_notes.apply(lambda x:audio_slicing(x, wav, sr, pad_before, pad_after), axis=1)
            track_notes['sampling_rate']=sr
            df_list.append(track_notes)
            if batching==True:
                if len(df_list)>(self.midi_wav_map.shape[0]/50):
                    create_df(df_list)
                    self.train, self.val, self.test = np.split(self.notes_collection.sample(frac=1, random_state=42),
                                [int(train*len(self.notes_collection)),
                                int((train+val)*len(self.notes_collection))])
                    self.train.to_pickle(os.path.join(dir_path, f"train_{self.batch_tracking}.pkl"))
                    self.val.to_pickle(os.path.join(dir_path, f"val_{self.batch_tracking}.pkl"))
                    self.test.to_pickle(os.path.join(dir_path, f"test_{self.batch_tracking}.pkl"))
                    print(f'saved batch {self.batch_tracking} data at {dir_path}')
                    self.batch_tracking=self.batch_tracking+1
                    df_list=[]
                    self.notes_collection=pd.DataFrame()
                    del self.train
                    del self.val
                    del self.test

                else:
                    pass
            else:
                pass
        create_df(df_list)
        
        

        if batching==True:
            self.train, self.val, self.test = np.split(self.notes_collection.sample(frac=1, random_state=42),
                                [int(train*len(self.notes_collection)),
                                int((train+val)*len(self.notes_collection))])
        else:
            self.notes_collection.to_pickle(os.path.join(dir_path, f"dataset_{self.batch_tracking}.pkl"))
        
        print('Done!')

        
        
        
    def augment_audio(self, audio_col='audio_wav', aug_col_names=None, aug_param_dict={}, train_only=False):
        """
        Apply audio augmentations to the training or full portion of a prepared audio dataset. The original dataset is modified to contain columns containing the augmented audio.
        
        Parameters:
            audio_col: String specifying the name of source audio column.
            aug_col_names: Names to use for augmented columns. Defaults to using the augmentation functions
                as column names.
            aug_param_dict: Dictionary of function names and associated parameters.
            train_only: Boolean for whether to augment the training set, or the data in its entirety.
        
        Example usage:
            data_container = data_preparation.data_preparation(gmd_path, dataset='gmd', sample_ratio=sample_ratio)
            data_container.augment_audio()
        """
        if not aug_param_dict:
            aug_param_dict = {
                'add_white_noise': {'snr':20, 'random_state': 42},
                'augment_pitch': {'n_steps':2, 'step_var':range(-1, 2, 1)},
                'add_pedalboard_effects': {}
            }
        if train_only:
            self.train = apply_augmentations(self.train, audio_col, aug_col_names, **aug_param_dict)
        else:
            self.notes_collection = apply_augmentations(self.notes_collection, audio_col, aug_col_names, **aug_param_dict)
