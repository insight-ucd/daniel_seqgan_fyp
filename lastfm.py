import numpy as np
import pandas as pd
from itertools import islice
import math
import time



pd.options.display.max_columns = 10000
pd.options.display.max_rows = 10000
import statistics


# # Read in .tsv file 
# # "lastfm/userid-timestamp-artid-artname-traid-traname.tsv"

# In[21]:

def read_data(nRows=None):
    start_time = time.time()

    df1 = pd.read_csv('/Users/dan/Desktop/FYP/lastfm/userid-timestamp-artid-artname-traid-traname.tsv', engine='python',
                      nrows=nRows, header=None, sep='\\t',
                      names=['userId', 'date', 'artist_id', 'artist_name', 'track_id', 'track_name'])
    df1.dataframeName = 'userid-timestamp-artid-artname-traid-traname.tsv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns, time taken {time.time()-start_time} seconds')

    df1.date = pd.to_datetime(df1.date)

    return df1


# # Give tracks unique id

# In[22]:

def assign_unique_ids(df1):
    daniel_track_id = {}
    start_time = time.time()

    for idx, value in enumerate(df1.track_name.unique()):
        daniel_track_id[value] = idx

    print(len(daniel_track_id))
    print("Completed in ", (time.time() - start_time), " seconds")

    return daniel_track_id

# # Function to create sessions from user listens

# In[23]:


def sessions_from_tracklist(daniel_track_id, tracks, df, session_size=25):
    count=0
    track_list = []
    track_times = []

    sessions = pd.DataFrame(columns=['user_session_ID', 'session', 'start_time', 'length'])
    
    session_size = np.timedelta64(session_size, 'm') # 30 minutes
    prev_time = np.timedelta64(0, 'us')

    for idx, row in df.iterrows():
        t_id = daniel_track_id[row.track_name]
        if t_id not in tracks:
            tracks[t_id] = {'track_name': row.track_name,  'artist': row.artist_name, 'listens': 1}
        else:
            tracks[t_id]['listens'] += 1
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


# # Group dataframe by users then call sessions_from_tracklist() for each group

# In[24]:


def sessions_from_frame(df1, daniel_track_id, session_size=25):
    start_time = st = time.time()
    all_tracks = {}
    session_data = pd.DataFrame()
    session_frame = pd.DataFrame(columns=['user_session_ID', 'session', 'start_time', 'length'])

    for idx, group in df1.groupby('userId')[['date', 'track_id', 'track_name', 'artist_name']]:
        group = group.sort_values(by='date', ascending=True)
        group.date = pd.to_timedelta(group.date.astype('int64'), unit='ns')
        session_data = sessions_from_tracklist(daniel_track_id, all_tracks, group, session_size)
        session_frame = session_frame.append(session_data, ignore_index=True)
        print(idx)
        print("session:\t{} total:\t{}".format(len(session_data), len(all_tracks)))
        print(" time taken:", time.time() - start_time)
        start_time=time.time()
        
    print("total time taken:", time.time() - st)
    print(type(session_frame.session.values.tolist()))
    # sorted track ids
    print([(k, (all_tracks[k]['artist'], all_tracks[k]['track_name'], all_tracks[k]['listens'])) for k in sorted(all_tracks, key=lambda x: all_tracks[x]['listens'], reverse=True)])

    return session_frame, all_tracks

def print_top_tracks(all_tracks, nTracks=10):
    sorted_tracks = [(k, (all_tracks[k]['artist'], all_tracks[k]['track_name'], all_tracks[k]['listens'])) for k in
                     sorted(all_tracks, key=lambda x: all_tracks[x]['listens'], reverse=True)]

    n_items = list(islice(sorted_tracks, 10))

    print("TOP TRACK\nRank\tArtist\t\t\tTrack\tListens")

    for idx, tk in enumerate(n_items):
        print(idx + 1, "  \t", tk[1][0], "  \t\t", tk[1][1], " \t", tk[1][2])

def find_repeated_tracks(session_frame):
    repeats = {}

    for row in session_frame.session:
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

    print(f"There are {repeats[2]} doubles, {repeats[3]} triples and {repeats[4]} quads")
    for key in repeats.keys():
        print(key, " - ", repeats[key])

# In[25]:

#if __name__ == '__main__':
def get_sessions(nRows):

    df = read_data(nRows)
    unique_ids = assign_unique_ids(df)
    session_frame, all_tracks = sessions_from_frame(df, unique_ids, session_size=20)
    print_top_tracks(all_tracks)
    df_new = session_frame.session
    print(df_new.head(10))
    print(df_new.shape)

    all_tracks.items
    tokens = []
    for row in df_new:
        tokens + row

    return session_frame.session, all_tracks


    # print(session_frame.head(10))
    # #for idx, row in session_frame[['user_session_ID', 'session', 'length']].iterrows():
    # for idx, row in session_frame.iterrows():
    #     print("usr:",row.user_session_ID[9:11], " sess:", row.user_session_ID[20:], " session_size:", len(row.session), " lenght:", math.floor(row.length.total_seconds()/60))










