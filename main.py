import argparse
from os import path
import os

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

    parser.add_argument('-km', '--kernel_mode',
                        choices=['performance', 'speed'],
                        type=str,
                        required=True,
                        help="The processing mode of the kernel, either speed or performance. "
                                "Speed mode is 4 times faster than performance mode but quality could be slightly worse")

    parser.add_argument('-bpm',
                        default=None,
                        type=int,
                        help='The estimated bpm of the song')

    parser.add_argument('-r', '--resolution',
                        default=16,
                        choices=[None, 4,8,16,32],
                        help='Control the window size (total length) of the onset sound clip extract from the song')

    parser.add_argument('-b', '--beat',
                        type=int,
                        default=4,
                        help='Number of beats in each measure')
        
    parser.add_argument('-n', '--note',
                        type=int,
                        default=4,
                        help="The UPPER NUMBER of the song's time signature." 
                                "This number represent the number of beats in each measure.")

    parser.add_argument('-fmt', '--format',
                        default='pdf',
                        choices=['pdf', 'musicxml'],
                        type=str,
                        help='Output sheet music format')
    
    parser.add_argument('-o', '--outpath',
                        default='',
                        type=str,
                        help='Output sheet music directory path')

    parser.add_argument('-on', '--outputfile_name',
                        default='Sheet Music',
                        type=str,
                        required=True,
                        help='Output sheet music file name, also serves as the srum sheet title')

    args = parser.parse_args()

    from inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
    from inference.transcriber import drum_transcriber
    from inference.prediction import predict_drumhit
    import librosa

    if args.link!=None:
        print(f'Downloading audio track from {args.link}')
        f_path = get_yt_audio(args.link)
        print(f'Audio track saved to {f_path}')
    else:
        f_path=args.path
        print(f'Retriving audio track from {args.path}')
    
    print('Start Demixing Process...')
    drum_track, sample_rate = drum_extraction(f_path,
                                              kernel=args.kernal,
                                              mode=args.kernel_mode)

    print('Drum track extracted')

    print('Converting drum track...')
    df, bpm = drum_to_frame(drum_track,
                            sample_rate,
                            estimated_bpm=args.bpm,
                            resolution=args.resolution)

    df_pred=predict_drumhit('inference/pretrained_models/annoteators/complete_network.h5', df, sample_rate)

    print('Creating sheet music...')

    song_duration = librosa.get_duration(drum_track, sr=sample_rate)

    sheet_music = drum_transcriber(df_pred,
                                    song_duration,
                                    bpm,
                                    sample_rate,
                                    beats_in_measure=args.beat,
                                    note_value=args.note,
                                    song_title=args.outputfile_name)
    
    if args.format=='pdf':
        out_path=sheet_music.sheet.write(fmt='musicxml.pdf', fp=os.path.join(args.outpath, args.outputfile_name))
        print(f'Sheet music saved at {out_path}')
    else:
        out_path= sheet_music.sheet.write(fp=os.path.join(args.outpath, args.outputfile_name))
        print(f'Sheet music saved at {out_path}')
    if args.link!=None:
        os.remove(f_path)
if __name__ == "__main__":
    main()