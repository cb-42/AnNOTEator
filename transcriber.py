import pandas as pd
import librosa
import numpy as np

class drum_transcriber():
    
    def __init__(self, prediction_df, song_duration, bpm, sample_rate, note_offset=None):
        
        self.bpm=bpm
        self.df=prediction_df
        self.sample_rate=sample_rate
        self.onsets=prediction_df.peak_sample
        self.note_line=self.onsets.apply(lambda x: librosa.samples_to_time(x, sr=sample_rate)).to_numpy()
        self.get_note_duration()

        if note_offset==None:
            total_8_note=[]
            for n in range(20):
                temp_8_div=self.get_eighth_note_time_grid(song_duration, note_offset=n)
                temp_synced_8_div=self.sync_8(temp_8_div)
                total_8_note.append(len(np.intersect1d(np.around(self.note_line,8), np.around(temp_synced_8_div,8))))
            note_offset=np.argmax(total_8_note)
        else:
            pass
        
        _8_div=self.get_eighth_note_time_grid(song_duration, note_offset=note_offset)
        self.synced_8_div=self.sync_8(_8_div)
        
        _16_div, _32_div, _8_triplet_div, _8_sixlet_div=self.get_note_division()
        
        self.synced_8_div_clean, self.synced_16_div, self.synced_32_div, self.synced_8_3_div, self.synced_8_6_div=self.master_sync(_16_div,
                                                                                                                              _32_div,
                                                                                                                              _8_triplet_div,
                                                                                                                              _8_sixlet_div)
    def get_note_duration(self):
        self._8_duration=60/self.bpm/2
        self._16_duration=60/self.bpm/4
        self._32_duration=60/self.bpm/8
        self._8_triplet_duration=self._8_duration/3
    
    def get_eighth_note_time_grid(self, song_duration, note_offset=0):
        first_note=librosa.samples_to_time(self.df.peak_sample.iloc[note_offset], sr=self.sample_rate)
        return np.arange(first_note, song_duration, self._8_duration)
    
    def sync_8(self, _8_div):
    
        #match timing of the first note
        synced_8_div=[_8_div[0]]
        diff_log=0

        #first, map and sync 8th notes to the onset  
        for note in _8_div[1:]:
            pos=np.argmin(np.abs(self.note_line-(note+diff_log)))
            diff=self.note_line[pos]-(note+diff_log)

            if np.abs(diff) > self._32_duration:
                synced_8_div.append(synced_8_div[-1]+self._8_duration)
            else:
                diff_log=diff_log+diff
                synced_8_div.append(note+diff_log)

        return np.array(synced_8_div)
    
    def get_note_division(self):
        
        _16_div=self.synced_8_div[:-1]+(np.diff(self.synced_8_div)/2)

        full_16_div=np.sort(np.concatenate((self.synced_8_div, _16_div), axis=0))
        _32_div=full_16_div[:-1]+(np.diff(full_16_div)/2)

        _8_triplet_a=self.synced_8_div[:-1]+(np.diff(self.synced_8_div)/3)
        _8_triplet_b=self.synced_8_div[:-1]+(np.diff(self.synced_8_div)/3*2)
        _8_triplet_div=np.sort(np.concatenate((_8_triplet_a, _8_triplet_b), axis=0))

        full_8_triple_div=np.sort(np.concatenate((_8_triplet_div, self.synced_8_div), axis=0))
        _8_sixlet_div=full_8_triple_div[:-1]+(np.diff(full_8_triple_div)/2)
        
        return _16_div, _32_div, _8_triplet_div, _8_sixlet_div
        
    def master_sync(self, _16_div, _32_div, _8_triplet_div, _8_sixlet_div):
        note_line_r=np.round(self.note_line,8)
        synced_eighth_r=np.round(self.synced_8_div,8)
        synced_16_div=[]
        synced_32_div=[]
        synced_8_3_div=[]
        synced_8_6_div=[]

        for i in range(len(synced_eighth_r) - 1):
            eighth_pair=synced_eighth_r[i:i+2]
            sub_notes=note_line_r[(note_line_r>eighth_pair[0]) & (note_line_r<eighth_pair[1])]
            if len(sub_notes)>0:
                dist_dict={'_16':[],'_32':[], '_8_3':[], '_8_6':[]}
                sub_notes_dict={'_16':np.round(np.linspace(self.synced_8_div[i], self.synced_8_div[i+1], 3),8)[:-1],
                                '_32':np.round(np.linspace(self.synced_8_div[i], self.synced_8_div[i+1], 5),8)[:-1],
                                '_8_3':np.round(np.linspace(self.synced_8_div[i], self.synced_8_div[i+1], 4),8)[:-1],
                                '_8_6':np.round(np.linspace(self.synced_8_div[i], self.synced_8_div[i+1], 7),8)[:-1]}

                for sub_note in sub_notes:
                    diff_16=np.min(np.abs(_16_div-sub_note))
                    dist_dict['_16'].append(diff_16)
                    _16closest_line=_16_div[np.argmin(np.abs(_16_div-sub_note))]
                    sub_notes_dict['_16'] = np.where(sub_notes_dict['_16'] == np.round(_16closest_line,8), sub_note, sub_notes_dict['_16'])

                    diff_32=np.min(np.abs(_32_div-sub_note))
                    dist_dict['_32'].append(diff_32)
                    _32closest_line=_32_div[np.argmin(np.abs(_32_div-sub_note))]
                    sub_notes_dict['_32'] = np.where(sub_notes_dict['_32'] == np.round(_32closest_line,8), sub_note, sub_notes_dict['_32'])

                    diff_8_triplet=np.min(np.abs(_8_triplet_div-sub_note))
                    dist_dict['_8_3'].append(diff_8_triplet)
                    _8_3closest_line=_8_triplet_div[np.argmin(np.abs(_8_triplet_div-sub_note))]
                    sub_notes_dict['_8_3'] = np.where(sub_notes_dict['_8_3'] == np.round(_8_3closest_line,8), sub_note, sub_notes_dict['_8_3'])

                    diff_8_sixlet=np.min(np.abs(_8_sixlet_div-sub_note))
                    dist_dict['_8_6'].append(diff_8_sixlet)
                    _8_6closest_line=_8_sixlet_div[np.argmin(np.abs(_8_sixlet_div-sub_note))]
                    sub_notes_dict['_8_6'] = np.where(sub_notes_dict['_8_6'] == np.round(_8_6closest_line,8), sub_note, sub_notes_dict['_8_6'])


                for key in dist_dict.keys():
                    dist_dict[key]=sum(dist_dict[key])/len(dist_dict[key])
                best_div=min(dist_dict, key=dist_dict.get)
                if best_div=='_16':
                    synced_16_div.extend(sub_notes_dict['_16'])
                elif best_div=='_32':
                    synced_32_div.extend(sub_notes_dict['_32'])
                elif best_div=='_8_3':
                    synced_8_3_div.extend(sub_notes_dict['_8_3'])
                else:
                    synced_8_6_div.extend(sub_notes_dict['_8_6'])

            else:
                pass

        #remove duplicate 8th
        synced_8_div_clean=self.synced_8_div.copy()
        for div in [synced_16_div, synced_32_div, synced_8_3_div, synced_8_6_div]:
            synced_8_div_clean=synced_8_div_clean[~np.in1d(np.around(synced_8_div_clean,8), np.around(div, 8))]
        return synced_8_div_clean, np.array(synced_16_div), np.array(synced_32_div), np.array(synced_8_3_div), np.array(synced_8_6_div)