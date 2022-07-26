import mido
from collections import Counter

#this file is not ready yet
note_map={
    35:'KD', 36: 'KD', 37: 'SD_xstick', 38: 'SD', 40: 'SD', 41: 'FT', 42: 'HH_close', 43: 'FT', 44: 'HH_close', 45: 'MT', 46: 'HH_open',
    47: 'MT', 48: 'HT', 49: 'CC', 50: 'HT', 51: 'RC', 52: 'CC', 53: 'RC', 54: 'HH_close', 55: 'CC', 79: 'CC', 56: 'CB', 57: 'CC', 59: 'RC'
 }


def midi_notes_extraction(path):       
    notes=[]
    midi=mido.MidiFile(path)
    for track in midi.tracks:
        for m in track:
            if m.type=='note_on' and m.velocity>0:
                if m.channel==9:
                    notes.append(m.note)
    notes_collector=Counter()
    notes_collector.update(notes)
    note_dict=dict(notes_collector)
    for key in note_dict.copy().keys():
        if key not in note_map:
            del note_dict[key]
        else:
            note_dict[note_map[key]] = note_dict.pop(key)
    return note_dict


def generate_test_data():
    test_data={
        '_21guns':{
            'note_count':midi_notes_extraction(r'song_midi\21guns.mid'),
            'yt_link':'https://www.youtube.com/watch?v=r00ikilDxW4'},
        # Smooth is a tricky song becuase it has a lot of other percussion that could not seperate out by the Spleeter package. They are all in the drum track, and our training sample never include percussion sample other than drum hit.
        'smooth':{
            'note_count':midi_notes_extraction(r'song_midi\Smooth.mid'),
            'yt_link':'https://www.youtube.com/watch?v=6Whgn_iE5uc'},
        'sweet_child_o_mine':{
            'note_count':midi_notes_extraction(r'song_midi\SweetChildOfMine.mid'),
            'yt_link':'https://www.youtube.com/watch?v=1w7OgIMMRc4'},
        # The dance of eternity from Dream Theatre is a rock song that has a BPM value at 122 and many notes were 32th notes, 
        # which means the duration of each 32th note is around 0.06 second only. Could be challenging for the model as well
        'the_dance_of_eternity':{
            'note_count':midi_notes_extraction(r'song_midi\TheDanceOfEternity.mid'),
            'yt_link':'https://www.youtube.com/watch?v=eYCYGpu0OxM'},
        'this_love':{
            'note_count':midi_notes_extraction(r'song_midi\ThisLove.mid'),
            'yt_link':'https://www.youtube.com/watch?v=XPpTgCho5ZA'},
        'if_i_aint_got_you':{
            'note_count':midi_notes_extraction(r'song_midi\IfIAintGotYou.mid'),
            'yt_link':'https://www.youtube.com/watch?v=Ju8Hr50Ckwk'},
    }
    return test_data