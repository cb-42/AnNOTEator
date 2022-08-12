import argparse
from os import path

def main():
    parser = argparse.ArgumentParser(description="Transcribe the drum part of a given song", usage=None)

    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument('-l', '--link',
                        type=str,
                        help='Youtube video link')
    
    input.add_argument('-p', '--path',
                        type=str,
                        help='Path to local audio file')

      
    parser.add_argument('-k', '--kernal',
                        default='demucs',
                        choices=['spleeter', 'demucs'],
                        type=str,
                        help='Kernel option to demix music')

    parser.add_argument('-ds', '--drum_start',
                        default=None,
                        type=int,
                        help='The start time (in seconds) of the drumplay in the song')

    parser.add_argument('-de', '--drum_end',
                        default=None,
                        type=int,
                        help='The end time (in seconds) of the drumplay in the song')

    parser.add_argument('-bpm',
                        default=None,
                        type=int,
                        help='The estimated bpm of the song')

    parser.add_argument('-r', '--resolution',
                        default=16,
                        choices=[None, 4,8,16,32],
                        help='Control the window size (total length) of the onset sound clip extract from the song')

    parser.add_argument('-bt', '--backtrack',
                        default=False,
                        type=bool,
                        help='Roll back the detected onset to the previous local minima')

    parser.add_argument('-fcl', '--fix_clip_length',
                        default=False,
                        type=bool,
                        help='Fix the clip length to 0.2 seconds')

    parser.add_argument('-b', '--beat',
                        type=int,
                        default=4,
                        help='Number of beats in each measure')
        
    parser.add_argument('-n', '--note',
                        type=int,
                        default=4,
                        help="The UPPER NUMBER of the song's time signature." 
                                "This number represent the number of beats in each measure.")

    parser.add_argument('-d', '--output_dir',
                        type=str,
                        required=True,
                        help="The LOWER NUMBER of the song's time signature." 
                            "This number represent the note value in a measure.")

    parser.add_argument('-f', '--outputfile_name',
                        type=str,
                        required=True,
                        help='The name of the output file')

    parser.add_argument('-fmt', '--format',
                        default='pdf',
                        choices=['pdf', 'musicxml'],
                        type=str,
                        help='Output sheet music format')

    args = parser.parse_args()

    from inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
    from inference.transcriber import drum_transcriber
    import librosa

    if args.link!=None:
        print(f'Downloading audio track from {args.link}')
        path = get_yt_audio(args.link)
        print(f'Audio track saved to {path}')
    else:
        path=args.path
        print(f'Retriving audio track from {args.path}')
    
    print('Start Demixing Process...')
    drum_track, sample_rate = drum_extraction(path,
                                              kernel=args.kernal,
                                              drum_start=args.drum_start,
                                              drum_end=args.drum_end)

    print('Drum track extracted')

    print('Converting drum track...')
    df, bpm = drum_to_frame(drum_track,
                            sample_rate,
                            estimated_bpm=args.bpm,
                            resolution=args.resolution,
                            backtrack=args.backtrack,
                            fixed_clip_length=args.fix_xlip_length)

    
    song_duration = librosa.get_duration(drum_track, sr=sample_rate)

    print('Creating sheet music...')
    sheet_music = drum_transcriber(df,
                                    song_duration,
                                    bpm,
                                    sample_rate,
                                    beats_in_measure=args.beat,
                                    note_value=args.note)
    
    if args.fmt=='pdf':
        filepath = path.join(args.output_dir, f'{args.outputfile_name}.pdf')
        sheet_music.sheet.write(fmt='musicxml.pdf', fp=filepath)
        print(f'Sheet music saved at {filepath}')
    else:
        filepath = path.join(args.output_dir, f'{args.outputfile_name}.mxml')
        sheet_music.sheet.write(fp=filepath)
        print(f'Sheet music saved at {filepath}')

if __name__ == "__main__":
    main()