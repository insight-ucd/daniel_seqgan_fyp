import numpy as np
import pandas as pd
from itertools import islice
import math
import time
import statistics



pd.options.display.max_columns = 10000
pd.options.display.max_rows = 10000


def read_data(nRows=None):
    start_time = time.time()
    directory = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
    df = pd.read_csv(directory, engine='python',
                          nrows=nRows, header=None, sep='\\t',
                          names=['userId', 'date', 'artist_id', 'artist_name', 'track_id', 'track_name'])
    df.dataframeName = 'userid-timestamp-artid-artname-traid-traname.tsv'
    df.date = pd.to_datetime(df.date)
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')

    return df

def summary_stats(df):
    nan_values = df.track_id.isna().sum()
    unique_tracks = len(df.track_name.unique())
    print("There are {} NaN track_ids and {} unique tracks".format(nan_values, unique_tracks))
    
def find_repeated_tracks(session_frame):
    repeats = {}

    for row in session_frame.iloc[1]:
        print(row)
        i = 0

        while i < (len(row) - 1):
            if row[i] == row[i + 1]:
                if 2 in repeats:
                    repeats[2] += 1
                else:
                    repeats[2] = 1
                j = i + 1
                count = 3
                while j < len(row) - 1:
                    if row[j] == row[j + 1]:
                        if count in repeats:
                            repeats[count] += 1
                        else:
                            repeats[count] = 1
                        j += 1
                        count += 1
                    else:
                        i = j
                        break
            i += 1
    return repeats    

def print_top_tracks(all_tracks, nTracks=10):
    sorted_tracks = [(k, (all_tracks[k]['artist'], all_tracks[k]['track_name'], all_tracks[k]['plays'])) for k in
                     sorted(all_tracks, key=lambda x: all_tracks[x]['plays'], reverse=True)]

    n_items = list(islice(sorted_tracks, 10))

    print("\nTOP TRACKS")
    print("{:<5s} {:<40s} {:<40s} {:<6s}".format("Rank","Artist","Track","Plays"))

    for idx, tk in enumerate(n_items):
        print("{:<5d} {:<40s} {:<40s} {:<6d} ".format(idx + 1, tk[1][0], tk[1][1], tk[1][2]))        

def assign_unique_ids(df1):
    print("Assigning unique track ids...")
    unique_track_id = {}
    start_time = time.time()

    for idx, value in enumerate(df1.track_name.unique()):
        unique_track_id[value] = idx

    print(len(unique_track_id))
    print("Completed in {:0.2f} seconds".format(time.time() - start_time))
    
    return unique_track_id


def sessions_from_tracks(daniel_track_id, tracks, df, session_size):
    count=0
    track_list = []
    track_times = []

    sessions = pd.DataFrame(columns=['user_session_ID', 'session', 'start_time', 'length'])
    
    session_size = np.timedelta64(session_size, 'm') 
    prev_time = np.timedelta64(0, 'us')

    for idx, row in df.iterrows():
        t_id = daniel_track_id[row.track_name]
  #  Calculate track plays       
        if t_id not in tracks:
            tracks[t_id] = {'track_name': row.track_name,  'artist': row.artist_name, 'plays': 1}
        else:
            tracks[t_id]['plays'] += 1
        track_list.append(daniel_track_id[row.track_name])
        track_times.append(row.date)
        
        
        if((row.date - prev_time)> session_size):
            userID = (row.userId + '_session_' + str(count+1))
            sessions = sessions.append({'user_session_ID': userID, 'session': track_list, 'start_time': track_times[0], 'length': (row.date - track_times[0])}, ignore_index=True)
            count += 1
            track_list = []
            track_times = []
        
        prev_time = row.date
    
    return sessions

def sessions_from_frame(df1, daniel_track_id, session_size):
    start_time = st = time.time()
    all_tracks = {}
    session_data = pd.DataFrame()
    session_frame = pd.DataFrame(columns=['user_session_ID', 'session', 'start_time', 'length'])

    for idx, group in df1.groupby('userId')[['date', 'track_id', 'track_name', 'artist_name']]:
        group = group.sort_values(by='date', ascending=True)
        group.date = pd.to_timedelta(group.date.astype('int64'), unit='ns')
        session_data = sessions_from_tracks(daniel_track_id, all_tracks, group, session_size)
        session_frame = session_frame.append(session_data, ignore_index=True)

        start_time=time.time()
      
    return session_frame, all_tracks

def get_sessions(nRows):

    df = read_data(nRows)
    summary_stats(df)
    unique_ids = assign_unique_ids(df)
    session_frame, all_tracks = sessions_from_frame(df, unique_ids, session_size=20)
    print_top_tracks(all_tracks)
    df_new = session_frame.session
    
    return df_new, all_tracks
