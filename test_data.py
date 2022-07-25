import mido
from collections import Counter

#this file is not ready yet

note_map={
33:'Metronome', 
34:	'Metronome',	 	 
35:	'KD',	 	 
36:	'KD',
60:	Hi Bongo,
37:	Side Stick,	 	
61:	Low Bongo,
38:	'SD',	 	
62:	Mute Hi Conga,
39:	Hand Clap,	 	
63:	Open Hi Conga,
40:	'SD',	 	
64:	Low Conga,
41:	Low Floor Tom,	 	
65:	High Timbale,
42:	Closed Hi-Hat,	 	
66:	Low Timbale,
43:	High Floor Tom,	 	
67:	High Agogo,
44:	Pedal Hi-Hat,	 	
68:	Low Agogo,
45:	Low Tom	 	A,	
69:	Cabasa,
46:	Open Hi-Hat,	 	
70:	Maracas,
47:	Low-Mid Tom,	 	
71:	Short Whistle,
48:	Hi-Mid Tom,	
72:	Long Whistle,
49:	'CC',	 	
73:	Short Guiro,
50:	High Tom, 	
74:	Long Guiro,
51:	'RC',	 	
75:	Claves,
52:	Chinese Cymbal,	 	
76:	Hi Wood Block,
53:	Ride Bell,	
77:	Low Wood Block,
54:	Tambourine,
78:	Mute Cuica,
55:	Splash Cymbal,	 
79:	Open Cuica,
56:	Cowbell,
80:	Mute Triangle,
57:	'CC',	 	
81:	Open Triangle,
58:	Vibraslap,		
59:	'RC'	 		
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
    return notes_collector

def generate_test_data():
    test_data={
        '_21guns':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\21guns.mid'),
            'yt_link':'https://www.youtube.com/watch?v=r00ikilDxW4'},
        'smooth':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\Smooth.mid'),
            'yt_link':'https://www.youtube.com/watch?v=6Whgn_iE5uc'},
        'sweet_child_o_mine':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\SweetChildOfMine.mid'),
            'yt_link':'https://www.youtube.com/watch?v=1w7OgIMMRc4'},
        'the_dance_of_eternity':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\TheDanceOfEternity.mid'),
            'yt_link':'https://www.youtube.com/watch?v=eYCYGpu0OxM'},
        'this_love':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\ThisLove.mid'),
            'yt_link':'https://www.youtube.com/watch?v=XPpTgCho5ZA'},
        'if_i_aint_got_you':{
            'note_count':midi_notes_extraction(r'C:\Users\Stanley\Desktop\song_midi\IfIAintGotYou.mid'),
            'yt_link':'https://www.youtube.com/watch?v=Ju8Hr50Ckwk'},
    }
    return test_data