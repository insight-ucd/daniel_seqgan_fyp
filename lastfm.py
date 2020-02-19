# -*- coding: utf-8 -*-
"""FYP_lastFM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZbPs90aJzvD-4gbZoDClyVP-bl-od1Op
"""

from google.colab import drive

drive.mount('/content/gdrive')

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

nRowsRead = 10000000
df1 = pd.read_csv('gdrive/My Drive/FYP/lastfm/userid-timestamp-artid-artname-traid-traname.tsv', engine='python', 
                  header=None, sep='\\t', nrows = nRowsRead,
                  names=['userId', 'date', 'artist_id', 'artist_name', 'track_id', 'track_name'])
df1.dataframeName = 'userid-timestamp-artid-artname-traid-traname.tsv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

df1.date = pd.to_datetime(df1.date)

!head -2 gdrive/My\ Drive/FYP/lastfm/userid-timestamp-artid-artname-traid-traname.tsv



pd.options.display.max_columns = 1000
pd.options.display.max_rows = None
df1.sort_values('userId', ascending=True).head(1000)

df1.groupby('userId').agg(len)

df1.groupby('userId')['date', 'track_id', 'track_name'].agg(list)

daniel_track_id = pd.DataFrame()
for idx, value in enumerate(df1.track_name.unique()):
  print(value," - ", idx)
  daniel_track_id[value] = idx

def sessions_from_tracklist(df):
    sessions = {}
    track_list = []
    count=0
    session_size = np.timedelta64(30, 'm') # 30 minutes
    prev_time = np.timedelta64(0, 'us')

    for idx, row in df.iterrows():
      track_list.append(row.track_id)

      if((row.date - prev_time)> session_size):
        sessions['session_' + str(count)] = track_list
        count +=1
        track_list = []
      
      prev_time = row.date
    # [ session_0:[t0, t1, ....], session_1: [], ...]
    return sessions

#session_data = pd.DataFrame()

for idx, group in df1.groupby('userId')[['date', 'track_id', 'track_name']]:
    print("\n",idx)
    group = group.sort_values(by='date', ascending=True)
    group.date = pd.to_timedelta(group.date.astype('int64'), unit='ns')
    session_data = sessions_from_tracklist(group)

    for key in session_data:
      print(session_data[key], end=' ')
    
    #print(idx, len(tracks), tracks[:len(tracks)])
    #print(pd.to_timedelta(group['date'].astype('int64'), unit='ns'))
    

# sequence length?
# cutoff interval for new session - 30mins currently 
# NaN values - replace with new id (hashed trackname) or use track_names

for idx, group in df1.groupby('userId')[['date']]:
  print(group.date)
  group.date = pd.to_timedelta(group.date.astype('int64'), unit='ns')

