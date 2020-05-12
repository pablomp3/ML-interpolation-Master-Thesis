'''For running this file it is necessary to install py-midi. There have been problems with it
(midi.read_midifile didn't exist) especially in last version of python (3.7)
I've managed to install it but I recommend python 3.6
py-midi from:
#install git+https://github.com/vishnubob/python-midi@feature/python3
#pip install git+https://github.com/vishnubob/python-midi@feature/python3
#you can try pip3 instead of pip as well

Functions midiToNoteStateMatrix and noteStateMatrixToMidi have been retrieved from
https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py?fbclid=IwAR2OyWkcUquqNHt8C73_f--ZtZfq69NWeh6O0TRtFAO0XeJB3GXHA30iubI
although modifications have been made in order to convert correctly 4/4 midi files from my data.
'''
import numpy as np
import pandas as pd
import midi
import os
import pickle

lowerBound = 36 #24
upperBound = 80 #102


def midiToNoteStateMatrix(midifile):
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound - lowerBound
    time = 0

    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    while True:
        #if time % (pattern.resolution / 4) == (pattern.resolution / 8):
        if time % (pattern.resolution / 2) == (pattern.resolution / 4):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            state[evt.pitch - lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix


def noteStateMatrixToMidi(statematrix, name="example"):
    statematrix = numpy.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperBound - lowerBound
    #tickscale = 55
    tickscale = 110

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

# check if a package is installed
'''import importlib
spam_spec = importlib.util.find_spec("py-midi")
found = spam_spec is not None
print(found)'''

folder_in = 'final_interpolation_dataset'
folder_out = 'training_matrices'

create_pickle_from_songs = False
if create_pickle_from_songs:
    directory = sorted(os.listdir(folder_in))[:]
    #print(directory[0:20])
    print(len(directory), "midi tracks to add to df and then save as pickle files by chunks")
    #print(directory)
    counter = []
    midi_names = []
    matrices = []
    shapes = []
    up_to_9999 = 0
    save_csv_name = 0 #int to change the name of the csv file every chunk
    i=0
    number_tracks_per_df = 9000 #size of the chunk #better if divisible by 3

    while(i<len(directory)):

        #print(i, " ", current_track_name, np.shape(curr_state_matrix), "-", np.shape(curr_state_matrix)[0])
        #if np.shape(curr_state_matrix)[0] != 43: #tracks who don't achieve 43
        #counter.append(np.shape(curr_state_matrix)[0])
        if len(directory)-i < number_tracks_per_df:
            remain = len(directory)-i
            print(remain, "files remaining")
            while (up_to_9999 < remain):
                current_track_name = directory[i]
                curr_state_matrix = midiToNoteStateMatrix(
                    folder_in + "/" + current_track_name)  # transform track to matrix for training
                midi_names.append(current_track_name)
                matrices.append(curr_state_matrix)
                shapes.append(np.shape(curr_state_matrix))
                up_to_9999 = up_to_9999 + 1
                i = i + 1
        else:
            while(up_to_9999 < number_tracks_per_df):
                current_track_name = directory[i]
                curr_state_matrix = midiToNoteStateMatrix(folder_in + "/" + current_track_name)  # transform track to matrix for training
                midi_names.append(current_track_name)
                matrices.append(curr_state_matrix)
                shapes.append(np.shape(curr_state_matrix))
                up_to_9999 = up_to_9999 + 1
                i=i+1
        up_to_9999 = 0
        d = {'midi_names':midi_names,'matrices':matrices, 'matrix_shape':shapes}
        df = pd.DataFrame(d)
        #df.to_csv(folder_out+'/matrices_'+str(save_csv_name)+'.csv', index=False)
        df.to_pickle(folder_out+'/matrices_pkl_'+str(save_csv_name)+'.pkl')
        midi_names = []
        matrices = []
        shapes = []
        print(i, "saved", save_csv_name, "as pickle")
        save_csv_name = save_csv_name + 1

# Merge all the csv/pickle files created
merge_pkl = True
files_to_merge = 3
merged_name = 'combined_3'
if merge_pkl:
    all_csv = sorted(os.listdir(folder_out))[:files_to_merge]
    print("files to merge:", all_csv)
    for i in range(0,len(all_csv)):
        all_csv[i] = folder_out+"/"+all_csv[i]
        #print(all_csv[i])
    print("location to merge:", all_csv)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_pickle(f) for f in all_csv])
    #export to csv/pickle
    #combined_csv.to_csv(folder_out+"/combined_csv.csv", index=False)
    combined_csv.to_pickle(folder_out+'/'+merged_name+'_pkl.pkl')
    print("all files already merged!")

prepare_clean_data = True
file_name = 'combined_clean_3'
if prepare_clean_data:
    # pad to add to tracks in order to reach a shape of 43x44x2
    pad = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    print('pad:', pad)
    print('pad shape', np.shape(pad))
    # pd.show_versions()
    object = pd.read_pickle('training_matrices/'+merged_name+'_pkl.pkl') # --- CHANGE NAME
    object = object.reset_index(drop=True)
    # object = object[:200]
    print('length of data:', len(object))
    remove_rows = []
    remove_values = []
    padded = []
    print(object[20875:])
    element = 0
    print('target length: 43')
    #print('track length:', np.shape(object[element:element + 1].iloc[0]['matrices'])[0])
    #print('track length:', object[element:element + 1].iloc[0]['matrix_shape'][0])

    for i in range(0,len(object)): # go through all tracks
        # if track length if above 43 or below 33 remove it from dataset
        if np.shape(object[i:i+1].iloc[0]['matrices'])[0] > 43 or np.shape(object[i:i+1].iloc[0]['matrices'])[0] < 33:
            remove_rows.append(i)
            remove_values.append(np.shape(object[i:i + 1].iloc[0]['matrices'])[0])

        # if track length is between 33 and 43 add pad to reach standard length of 43
        if np.shape(object[i:i+1].iloc[0]['matrices'])[0] >= 33 and np.shape(object[i:i+1].iloc[0]['matrices'])[0] < 43:
            while (np.shape(object[i:i+1].iloc[0]['matrices'])[0] < 43):
                object[i:i + 1].iloc[0]['matrices'].append(pad)
            padded.append(np.shape(object[i:i + 1].iloc[0]['matrices'])[0])

    print(len(padded), 'tracks restored (padded)')
    print(remove_rows)
    print(remove_values)
    print(len(remove_rows), 'tracks to remove:', remove_rows)
    object = object.drop(index = remove_rows, axis=0)
    print(len(remove_values), 'tracks removed:')
    print('length of data after cleaning:', len(object))

    # confirmation of successfully deletion (aka all tracks must have length of 43)
    remove_values_1 = []
    remove_rows_1 = []
    for i in range(0,len(object)):
        if np.shape(object[i:i + 1].iloc[0]['matrices'])[0] != 43:
            remove_rows_1.append(i)
            remove_values_1.append(np.shape(object[i:i + 1].iloc[0]['matrices'])[0])
    print('to remove (should be empty if correct):', remove_values_1)
    if len(remove_rows_1) > 0:
        object = object.drop(remove_rows)

    print('length of data after second cleaning:', len(object))
    # delete names and shapes of the dataframe
    object.drop(columns='midi_names', inplace=True)
    object.drop(columns='matrix_shape', inplace=True)
    object.to_pickle(folder_out+'/'+file_name+'_pkl.pkl')
    object.to_csv(folder_out+'/'+file_name+'_csv.csv', index=False)