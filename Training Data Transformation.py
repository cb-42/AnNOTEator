#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:


# load the resampled data

# this notebook is the demonstration of how we transform the resampled data into
# 4d numpy array which can be directly fit to Keras network for training

# the output dataframe we got from data_preparation script, after the resampling step
# this is a sample that contains only 100 instances for this demonstration

df = pd.read_pickle('gmd_100sample.pkl')
df[['audio_wav_resample', 'resample_sr', 'label']].head()


# In[3]:


# label encoding            
# some instances there are multiple positive labels

mlb = MultiLabelBinarizer()
y = pd.DataFrame(mlb.fit_transform(df['label']),columns=mlb.classes_)
y


# In[4]:


# in the groove set there are 31 different labels


# In[5]:


# select only the the most common labels to model for now  
# if we can find some solution(s) that work out well        then we expand the labels and data

cols = ['D','S','H','_','e','o','c','s','l','K','C']
label = y[cols]

# since we leave out some labels     certain instances will have "no label" no
# we encode these as "other"

# (this is one of the things that are uncertain choices, this is the usual way to do when we leave out some labels)
# we can also ...   
# 1. discard the no-label instances
# 2. just leave them as a bunch of zeros       [0,0,0..., 0]

# this will be tested and we just take what turns out to work best since each way have it's own logic

label['sum'] = label[cols].sum(axis = 1)

def other_label(label_sum):
    if label_sum == 0:
        return 1
    else: 
        return 0

label['other'] = label.apply(lambda row: other_label(row['sum']),axis=1)
label.drop(columns=['sum'],inplace=True)

label.head()


# In[6]:


# about 13.6% of the instances in the gmd set are labeled as "other",
# namely, non of the top 11 labels appeared in these instances


# In[7]:


# turn the resampled autio_wav into the representation of short-time fourier transformation (stft)

stft_train = []

for i in range(df.shape[0]):
    stft_train.append(np.abs(librosa.stft(df.audio_wav_resample.iloc[i])))

X1 = np.array(stft_train)
X1 = X1.reshape(X1.shape[0],X1.shape[1],X1.shape[2],1)
X1.shape
    
# this is in the format of 
# a 4d array of (number of instances, y axis shape, x axis shape, 1 channel)


# In[ ]:


# 100 stft representation of 100 instances, each has the shape (1025,11), has 1 channel

# (normal image classification tasks the instances will have multiple channels)
# the task we're doing is audio classification, which will only have 1 channel for all representation formats


# In[8]:


# turn the resampled autio_wav into the representation of Mel-Frequency Spectrogram 

mel_train = []

for i in range(df.shape[0]):
    mel_train.append(librosa.feature.melspectrogram(y=df.audio_wav_resample.iloc[i], 
                                                    sr=df.resample_sr.iloc[i], 
                                                    n_mels=128, fmax=8000))

X2 = np.array(mel_train)
X2 = X2.reshape(X2.shape[0],X2.shape[1],X2.shape[2],1)
X2.shape


# In[9]:


# turn the resampled autio_wav into the representation of Mel-Frequency Cepstral Coefficients

mfcc_train = []

for i in range(df.shape[0]):
    mfcc_train.append(librosa.feature.mfcc(y=df.audio_wav_resample.iloc[i], 
                                           sr=df.resample_sr.iloc[i]         
                                           ))                            

X3 = np.array(mfcc_train)
X3 = X3.reshape(X3.shape[0],X3.shape[1],X3.shape[2],1)

X3.shape


# In[10]:


y = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

x_train.shape, y_train.shape

