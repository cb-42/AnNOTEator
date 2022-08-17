import pandas as pd
import numpy as np
from tensorflow import keras
import librosa

def predict_drumhit(network,df, song_sampling_rate):

  '''
  :param network (file path):           Path to the trained keras network
  :param df (Pandas DataFrame):         The output dataframe from drum_to_frame function 
  :param song_sampling_rate (int):      The sampling rate of the song

  :return result (Pandas DataFrame):    The dataframe with prediction labels
  '''

  model = keras.models.load_model(network)
  
  pred_x = []

  for i in range(df.shape[0]):
    pred_x.append(librosa.feature.melspectrogram(y=df.audio_clip.iloc[i], 
                                                 sr=song_sampling_rate, n_mels=128, fmax=8000))

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
