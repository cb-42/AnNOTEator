import mido
from mido import MidiFile, Message, MidiTrack, MetaMessage
from IPython.display import Audio
import librosa
import librosa.display
import pandas as pd
import os
from tqdm.notebook import tqdm
import numpy as np
import itertools



class data_preparation():
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
    
    :raise NameError: the use of other dataset is not supported
    """
    
    def __init__(self, directory_path, dataset, sample_ratio=1):
        if dataset in ['gmd', 'egmd']:
            self.directory_path=directory_path
            self.dataset_type=dataset
            csv_path=[f for f in os.listdir(directory_path) if '.csv' in f][0]
            self.dataset=pd.read_csv(os.path.join(directory_path, csv_path)).dropna().sample(frac=sample_ratio).reset_index()
            self.midi_wav_map=self.dataset[['index', 'midi_filename', 'audio_filename']]
            self.midi_wav_map.columns=['track_id', 'midi_filename', 'audio_filename']
            self.notes_collection=pd.DataFrame()
            # the midi note mapping is copied from the Google project page. note 39,54,56 are new in 
            # EGMD dataset and Google never assigned it to a code. From initial listening test, these
            # are electronic pad / cowbell sound, temporaily assigned it to CB, CowBell for now
            self.midi_note_map={36:'KD', 38:'SD', 40:'SD', 37:'SD_xstick', 48:'HT', 50:'HT',
                               45:'MT', 47:'MT', 43:'FT' ,58:'FT', 46:'HH_open', 
                               26:'HH_open', 42:'HH_close', 22:'HH_close', 44:'HH_close',
                               49:'CC', 57:'CC', 55:'CC', 52:'CC', 51:'RC',
                               59:'RC', 53:'RC', 39:'CB', 54:'CB', 56:'CB'}
            
        else:
            raise NameError('dataset not supported')
    
    def notes_extraction(self, midi_file):
        """
        A function to extract notes from the miditrack. A miditrack is composed by a series of MidiMessage.
        Each MidiMessage contains information like channel, note, velocity, time etc.
        This function extract each note presented in the midi track and the associated start and stop playing time of each note
        
        :param midi_file (string): The midi file path
        :return (list): a list of 3-element lists, each 3-element list consist of [note label, associated start time in the track, associated end time in the track]  
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
                    notes_collection.append([msg.note, round(temp_dict[msg.note][0],2), round(end,2)])
                    del temp_dict[msg.note][0]
                    if len(temp_dict[msg.note])==0:
                        del temp_dict[msg.note]
                else:
                    notes_collection.append([msg.note, round(temp_dict[msg.note],2), round(end,2)])
                    del temp_dict[msg.note]

            else:
                continue
        return [[self.midi_note_map[n[0]], n[1], n[2]] for n in notes_collection]
    
    def time_meta_extraction(self):
        """
        A helper function to extract ticks_per_beat and tempo information from the meta messages in the midi file. 
        These information are required to convert midi time into seconds.
        """
        for msg in self.midi_track.tracks[0]:
            if msg.type=='time_signature':
                ticks_per_beat=msg.notated_32nd_notes_per_beat
            elif msg.type=='set_tempo':
                tempo=msg.tempo
            else:
                pass
        return (ticks_per_beat, tempo)
        
        
    def merge_note_label(self, track_id, notes_collection):
        """
        Merge the notes if they share the same start time, which means these notes were start playing at the same time.
        
        :param track_id: the unique id of the midi_track
        :param notes_collection: the list of 3-element lists, each 3-element list consist of
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
        
        :param notes_collection: the list of 3-element lists, each 3-element list consist of
                                                        [note label, associated start time in the track, associated end time in the track]
        :return: a list of 3-element lists with start and end time rounded to 2 decimal places in seconds
        """
        if type(self.time_log)==float:
            return notes_collection
        else:
            ticks_per_beat, tempo=self.time_meta_extraction()

            return [[note[0],
                     round(mido.tick2second(note[1],ticks_per_beat,tempo)/60,2),
                     round(mido.tick2second(note[2],ticks_per_beat,tempo)/60,2) ] for note in notes_collection]
    

        
    def create_audio_set(self, padding_=0.02, train=0.6, val=0.2, test=0.2, random_state=42):
        """
        main function to create training/test/eval dataset from dataset
        
        :param padding_: padding (in seconds) add to both end of the sliced audio. default 0.02 seconds
                         The padding actually increas the window length when doing the slicing instead of adding white space before and after.
        :param train, val, test: the train/val/test ratio
        :param random_state: random_state, default 42
        """
        tqdm.pandas()
        df_list=[]
        
        if train+val+test!=1: raise ValueError('the total of train, val, test should euqal to 100%') 
        
        def audio_slicing(x, wav, sr, padding_):
            padding=librosa.time_to_samples(padding_, sr=sr)
            start=max(librosa.time_to_samples(x['start'], sr=sr)-padding, 0)
            end=min(librosa.time_to_samples(x['end'], sr=sr)+padding, len(wav))
            return wav[start:end]
        def resampling(x, max_length):
            org_sr=x['sampling_rate']
            tar_sr_ratio=max_length/len(x['audio_wav'])
            return pd.Series([librosa.resample(x['audio_wav'], orig_sr=org_sr, target_sr=int(org_sr*tar_sr_ratio)), int(org_sr*tar_sr_ratio)])


            
        for key, row in tqdm(self.midi_wav_map.iterrows(), total=self.midi_wav_map.shape[0]):
            if row['midi_filename']=='drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
                continue
            notes_collection=self.notes_extraction(row['midi_filename'])
            converted_notes_collection=self.ticks_to_second(notes_collection)
            track_notes=self.merge_note_label(row['track_id'], converted_notes_collection)
            
        
            wav, sr = librosa.load(os.path.join(self.directory_path, row['audio_filename']))
            track_notes['audio_wav']=track_notes.apply(lambda x:audio_slicing(x, wav, sr, padding_), axis=1)
            track_notes['sampling_rate']=sr
            df_list.append(track_notes)
                       
        self.notes_collection=pd.concat(df_list, ignore_index=True)
        self.notes_collection=self.notes_collection[self.notes_collection['audio_wav'].apply(lambda x:len(x))!=0]
        max_length=self.notes_collection['audio_wav'].apply(lambda x:len(x)).max()
        self.notes_collection[['audio_wav_resample', 'resample_sr']]=self.notes_collection.progress_apply(lambda x:resampling(x, max_length), axis=1)

        self.train, self.val, self.test = np.split(self.notes_collection.sample(frac=1, random_state=random_state),
                                         [int(train*len(self.notes_collection)),
                                          int((train+val)*len(self.notes_collection))])