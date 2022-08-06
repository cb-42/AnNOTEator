import argparse
# import librosa
import pandas as pd
import numpy as np
from tensorflow import keras
import sys
    
    
def main():
    
    # Define and parse arguments
    parser = argparse.ArgumentParser(description="AnNOTEators' Inference Script")
    parser.add_argument("-s", "--spectrogram", help="Spectrogram filepath; a Mel spectrogram in Numpy array format.")
    parser.add_argument("-m", "--metadata", help="Metadata filepath; additional metadata to use in sheet music contruction.")    
    parser.add_argument("-n", "--network", help="Network filepath; pre-trained Keras model.")
    parser.add_argument("-o", "--output", help="Output filepath; predictions in Pandas DataFrame format.")
    args = parser.parse_args()
#     print('Args:', args)
              
    
    # retrieve mel processed input
    mel_input = np.load(args.spectrogram)
    
    # load model, predict and format predictions
    model = keras.models.load_model(args.network)
    pred = np.round(model.predict(mel_input))
    drum_hits = ['SD','HH_close','KD','RC','FT','HT','HH_open','SD_xstick','MT','CC']
    pred_df = pd.DataFrame(pred, columns=drum_hits)
    # Store result
    pred_df.to_csv(args.output, index=False)
    
    if args.metadata:
        pass # TBD

if __name__ == "__main__":
    main()
