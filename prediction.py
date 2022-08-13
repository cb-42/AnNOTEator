import pandas as pd
import numpy as np
from tensorflow import keras
import librosa
from input_transform import drum_extraction, drum_to_frame, get_yt_audio


def predict_drumhit(network,yt_link=True,path=None,est_bpm=None)

  '''
  parameter network: path to the trained keras network
  parameter yt_link: boolean, extracting music from youtube link or not
  parameter path: if yt_link = True, input the youtube link here; else, input the local audio filepath
  parameter est_bpm: the estimated bpm for the music

  Examples: 
  yt_music_usage : predict_drumhit('Trained Network/trained_network_path', yt_link=True,
                                   path = 'https://www.youtube.com/watch?v=Y7ix6RITXM0',est_bpm=110)
  local_input_usage : predict_drumhit('Trained Network/trained_network_path', yt_link=False, 
                                      path = 'content/drums.wav',est_bpm=96)
  '''

  model = keras.models.load_model(network)

  if yt_link = True:
    path = get_yt_audio(path)
    drum_track, sr = drum_extraction(path, kernel='demucs')

  else :
    drum_track, sr = librosa.load(path)

  df, bpm=drum_to_frame(drum_track,sample_rate=sr,
            hop_length=1024,
            backtrack=False,
            estimated_bpm=est_bpm,
            fixed_clip_length=False,
            resolution=16)
  
  pred_x = []

  for i in range(df.shape[0]):
    pred_x.append(librosa.feature.melspectrogram(y=df.audio_clip.iloc[i], 
                          sr=df.sampling_rate.iloc[i], n_mels=128, fmax=8000))

  X = np.array(pred_x)
  X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)


  result = []
  pred_raw = model.predict(X)
  pred = np.round(pred_raw)

  for i in range(pred_raw.shape[0]):
    prediction = pred[i]
    if sum(prediction) == 0:
      raw = pred_raw[i]
      new = np.zeros(6)
      ind = raw.argmax()
      new[ind] = 1
      result.append(new)
    else:
      result.append(prediction)

  result = np.array(result)

  drum_hits = ['SD','HH','KD','RC','TT','CC']
  prediction = pd.DataFrame(result, columns = drum_hits)

  df.reset_index(inplace=True)
  prediction.reset_index(inplace=True)

  result = df.merge(prediction,left_on='index', right_on= 'index')
  result.drop(columns=['index'],inplace=True)
  
  return result