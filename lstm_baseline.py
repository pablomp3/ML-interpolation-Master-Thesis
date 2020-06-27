import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
#from flow import shuffle_in_unison_scary
#import analyze
import models
import tensorflow
from midi_state_conversion import midiToNoteStateMatrix
from midi_state_conversion import noteStateMatrixToMidi
from midi_state_conversion import padStateMatrix
# univariate lstm example
import keras
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Dropout
pad_64x2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
build_data = False
build_data_2 = False
build_data_unseen_test = True
load_data = True
training = True
evaluating = True

target_length = 64
pad = pad_64x2
if build_data:
    #folder_name = 'final_interpolation_dataset_flow_test'
    folder_name = 'final_interpolation_dataset_full_encoded'
    directory = sorted(os.listdir(folder_name))[:]
    feed_in = []
    print("total len:", len(directory))
    for i in range(0,len(directory)-2):
        song_name_1 = directory[i]
        song_name_2 = directory[i+1]
        song_name_3 = directory[i+2]
        #print(song_name)
        track_type = song_name_1[-6:-5]
        #print(track_type)
        if i%3==0:
            print(i, song_name_1, song_name_2, song_name_3)
            state_1 = midiToNoteStateMatrix(folder_name + '/' + song_name_1)  # shape 43x64x2
            state_2 = midiToNoteStateMatrix(folder_name + '/' + song_name_2)  # shape 43x64x2
            state_3 = midiToNoteStateMatrix(folder_name + '/' + song_name_3)  # shape 43x64x2
            state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
            state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
            state_3 = padStateMatrix(folder_name, song_name_3, target_length, pad, save_as_midi=False)  # shape 64x64x2
            state_1 = state_1[0:42]
            state_2 = state_2[0:42]
            state_3 = state_3[0:42]

            state_1 = state_1.astype(np.float32)
            state_2 = state_2.astype(np.float32)
            state_3 = state_3.astype(np.float32)

            full_track = np.concatenate((state_1,state_2), axis=0)
            print("full_track", full_track.shape)
            full_track = np.concatenate((full_track,state_3), axis=0)
            print("full_track final", full_track.shape) # 2 tracks of 42x64x2 each = 126x64x2
            feed_in.append(full_track)
        print((i+1)*100/(len(directory)-2),"% songs processed")
    feed_in = np.asarray(feed_in)
    feed_in = feed_in.astype(np.float32)
    print("feed_in final", feed_in.shape)  # 2001 tracks of 126x64x2 each
    np.random.shuffle(feed_in)
    print("after shuffle feed_in final", feed_in.shape)  # 2001 tracks of 126x64x2 each
    np.save('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm.npy', feed_in)  # save

if build_data_unseen_test:
    #folder_name = 'final_interpolation_dataset_flow_test'
    folder_name = 'final_interpolation_dataset_unseen'
    directory = sorted(os.listdir(folder_name))[:3]
    test_in = []
    print("total len:", len(directory))
    for i in range(0,len(directory)-2):
        song_name_1 = directory[i]
        song_name_2 = directory[i+1]
        song_name_3 = directory[i+2]
        #print(song_name)
        track_type = song_name_1[-6:-5]
        #print(track_type)
        if i%3==0:
            print(i, song_name_1, song_name_2, song_name_3)
            state_1 = midiToNoteStateMatrix(folder_name + '/' + song_name_1)  # shape 43x64x2
            #state_2 = midiToNoteStateMatrix(folder_name + '/' + song_name_2)  # shape 43x64x2
            state_3 = midiToNoteStateMatrix(folder_name + '/' + song_name_3)  # shape 43x64x2
            state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
            #state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
            state_3 = padStateMatrix(folder_name, song_name_3, target_length, pad, save_as_midi=False)  # shape 64x64x2
            state_1 = state_1[0:42]
            state_1_last = state_1[0:42]
            #state_2 = state_2[0:42]
            state_3 = state_3[0:42]
            state_3_last = state_3[0:42]

            state_1 = state_1.astype(np.float32)
            #state_2 = state_2.astype(np.float32)
            state_3 = state_3.astype(np.float32)

            #full_track = np.concatenate((state_1,state_2), axis=0)
            #print("full_track", full_track.shape)
            #full_track = np.concatenate((full_track,state_3), axis=0)
            #print("full_track final", full_track.shape) # 2 tracks of 42x64x2 each = 126x64x2
            #test_in.append(full_track)
        print((i+1)*100/(len(directory)-2),"% unseen test songs processed")
        state_test = state_1[39:]
        print("length of state_test", state_test.shape)

if build_data_2:
    begin_arr = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm.npy')  # load
    print(type(begin_arr), begin_arr.shape)
    limit = int(0.8 * len(begin_arr))
    begin_arr_train = begin_arr[:limit]
    print("ARR TRAIN:", begin_arr_train.dtype, begin_arr_train.shape)
    begin_arr_test = begin_arr[limit:]
    print("ARR TEST:", begin_arr_train.dtype, begin_arr_train.shape)

    X = []  # array containing sequences of begin and end (input)
    X_test = []  # array containing sequences of begin and end (input)
    y = []  # array containing sequences of interpolation (output)
    y_test = []  # array containing sequences of interpolation (output)

    print("first example of training arr:", len(begin_arr_train[0]))  # length of each track = 126
    print("Building X and y to feed the LSTM")
    for k in range(0, len(begin_arr_train)):  # 533 (number of songs)
        for i in range(0, len(begin_arr_train[0])):  # 42 (length of each track)
            end_index = i + 3
            if end_index > len(begin_arr_train[0]) - 1:
                break

            begin_timestep_vector_1 = begin_arr_train[0][i]  # vector 64x2 of first timestep of begin track
            #print("vector_i", begin_timestep_vector_1.shape)
            begin_timestep_vector_2 = begin_arr_train[0][i + 1]  # vector 64x2 of first timestep of begin track
            #print("vector_ii", begin_timestep_vector_2.shape)
            begin_timestep_vector_3 = begin_arr_train[0][i + 2]  # vector 64x2 of first timestep of begin track
            #print("vector_iii", begin_timestep_vector_3.shape)

            seq_x = np.dstack((begin_timestep_vector_1, begin_timestep_vector_2))
            #print("seq_x:", seq_x.shape)
            seq_x = np.dstack((seq_x, begin_timestep_vector_3))  # shape 64x2x3 -> 3x64x2 (using dstack)
            seq_x = np.einsum('ijk->kij', seq_x)  # shape 3x64x2
            #print("final seq_x:", seq_x.shape)
            seq_x = seq_x.reshape((3, 64 * 2))
            #print("final seq_x:", seq_x.shape)

            inter_timestep_vector = begin_arr_train[0][end_index]  # vector 64x2 of first timestep of interpolating track
            seq_y = inter_timestep_vector
            seq_y = seq_y.reshape((64 * 2))
            #print("final seq_y:", seq_y.shape)

            X.append(seq_x)
            y.append(seq_y)
        print("X,y_train", (k + 1) * 100 / (len(begin_arr_train)), "% songs saved")
    print(np.asarray(X).shape)
    print(np.asarray(y).shape)
    X = np.asarray(X)
    y = np.asarray(y)
    np.save('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_X.npy', X)  # save
    np.save('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_y.npy', y)  # save

    print("first example of testing arr:", len(begin_arr_test[0]))  # length of each track = 126
    print("Building X_test and y_test to feed the LSTM")
    for k in range(0, len(begin_arr_test)):  # 533 (number of songs)
        for i in range(0, len(begin_arr_test[0])):  # 42 (length of each track)
            end_index = i + 3
            if end_index > len(begin_arr_test[0]) - 1:
                break

            begin_timestep_vector_1 = begin_arr_test[0][i]  # vector 64x2 of first timestep of begin track
            # print("vector_i", begin_timestep_vector_1.shape)
            begin_timestep_vector_2 = begin_arr_test[0][i + 1]  # vector 64x2 of first timestep of begin track
            # print("vector_ii", begin_timestep_vector_2.shape)
            begin_timestep_vector_3 = begin_arr_test[0][i + 2]  # vector 64x2 of first timestep of begin track
            # print("vector_iii", begin_timestep_vector_3.shape)

            seq_x = np.dstack((begin_timestep_vector_1, begin_timestep_vector_2))
            # print("seq_x:", seq_x.shape)
            seq_x = np.dstack((seq_x, begin_timestep_vector_3))  # shape 64x2x3 -> 3x64x2 (using dstack)
            seq_x = np.einsum('ijk->kij', seq_x)  # shape 3x64x2
            # print("final seq_x:", seq_x.shape)
            seq_x = seq_x.reshape((3, 64 * 2))
            # print("final seq_x:", seq_x.shape)

            inter_timestep_vector = begin_arr_test[0][end_index]  # vector 64x2 of first timestep of interpolating track
            seq_y = inter_timestep_vector
            seq_y = seq_y.reshape((64 * 2))
            # print("final seq_y:", seq_y.shape)

            X_test.append(seq_x)
            y_test.append(seq_y)
        print("X,y_test", (k + 1) * 100 / (len(begin_arr_train)), "% songs saved")
    print(np.asarray(X_test).shape)
    print(np.asarray(y_test).shape)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    np.save('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_X_test.npy', X_test)  # save
    np.save('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_y_test.npy', y_test)  # save

if load_data:
    # LOAD FULL DATASET (processing is slow to I saved it first)
    '''
    X = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_X.npy')
    y = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_y.npy')
    X_test = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_X_test.npy')
    y_test = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm_y_test.npy')
    print("X_train", type(X), X.shape) # (3033672, 3, 128)
    print("y_train", type(y), y.shape) # (3033672, 128)
    '''

    # PROCESS AND LOAD PARTIAL DATASET (very quick)
    #'''
    begin_arr = np.load('final_interpolation_dataset_np' + '/TOTAL_full_songs_for_lstm.npy')  # load little data, len = 667  (667, 126, 64, 2)
    begin_arr = begin_arr[:66700,:,:,:]
    print(type(begin_arr), begin_arr.shape)
    limit = int(0.8 * len(begin_arr))
    #begin_arr_train = begin_arr[:limit]
    begin_arr_train = begin_arr[:limit,:,:,:]
    print("ARR TRAIN:", begin_arr_train.dtype, begin_arr_train.shape)
    begin_arr_test = begin_arr[limit:,:,:,:]
    print("ARR TEST:", begin_arr_test.dtype, begin_arr_test.shape)

    X = []  # array containing sequences of begin and end (input)
    X_test = []  # array containing sequences of begin and end (input)
    y = []  # array containing sequences of interpolation (output)
    y_test = []  # array containing sequences of interpolation (output)

    print("first example of training arr:", len(begin_arr_train[0]))  # length of each track = 126
    print("Building X and y to feed the LSTM")
    for k in range(0, len(begin_arr_train)):  # 533 (number of songs)
        for i in range(0, len(begin_arr_train[0])):  # 42 (length of each track)
            end_index = i + 3
            if end_index > len(begin_arr_train[0]) - 1:
                break

            begin_timestep_vector_1 = begin_arr_train[0][i]  # vector 64x2 of first timestep of begin track
            # print("vector_i", begin_timestep_vector_1.shape)
            begin_timestep_vector_2 = begin_arr_train[0][i + 1]  # vector 64x2 of first timestep of begin track
            # print("vector_ii", begin_timestep_vector_2.shape)
            begin_timestep_vector_3 = begin_arr_train[0][i + 2]  # vector 64x2 of first timestep of begin track
            # print("vector_iii", begin_timestep_vector_3.shape)

            seq_x = np.dstack((begin_timestep_vector_1, begin_timestep_vector_2))
            # print("seq_x:", seq_x.shape)
            seq_x = np.dstack((seq_x, begin_timestep_vector_3))  # shape 64x2x3 -> 3x64x2 (using dstack)
            seq_x = np.einsum('ijk->kij', seq_x)  # shape 3x64x2
            # print("final seq_x:", seq_x.shape)
            seq_x = seq_x.reshape((3, 64 * 2))
            # print("final seq_x:", seq_x.shape)

            inter_timestep_vector = begin_arr_train[0][
                end_index]  # vector 64x2 of first timestep of interpolating track
            seq_y = inter_timestep_vector
            seq_y = seq_y.reshape((64 * 2))
            # print("final seq_y:", seq_y.shape)

            X.append(seq_x)
            y.append(seq_y)
        print("X,y_train_mini", (k + 1) * 100 / (len(begin_arr_train)), "% little songs processed")
    print("X_train_mini", type(X), np.asarray(X).shape)
    print("y_train_mini", type(y), np.asarray(y).shape)
    X = np.asarray(X)
    y = np.asarray(y)
    print("first example of testing arr:", len(begin_arr_test[0]))  # length of each track = 126
    print("Building X_test and y_test to feed the LSTM")
    for k in range(0, len(begin_arr_test)):  # 533 (number of songs)
        for i in range(0, len(begin_arr_test[0])):  # 42 (length of each track)
            end_index = i + 3
            if end_index > len(begin_arr_test[0]) - 1:
                break

            begin_timestep_vector_1 = begin_arr_test[0][i]  # vector 64x2 of first timestep of begin track
            # print("vector_i", begin_timestep_vector_1.shape)
            begin_timestep_vector_2 = begin_arr_test[0][i + 1]  # vector 64x2 of first timestep of begin track
            # print("vector_ii", begin_timestep_vector_2.shape)
            begin_timestep_vector_3 = begin_arr_test[0][i + 2]  # vector 64x2 of first timestep of begin track
            # print("vector_iii", begin_timestep_vector_3.shape)

            seq_x = np.dstack((begin_timestep_vector_1, begin_timestep_vector_2))
            # print("seq_x:", seq_x.shape)
            seq_x = np.dstack((seq_x, begin_timestep_vector_3))  # shape 64x2x3 -> 3x64x2 (using dstack)
            seq_x = np.einsum('ijk->kij', seq_x)  # shape 3x64x2
            # print("final seq_x:", seq_x.shape)
            seq_x = seq_x.reshape((3, 64 * 2))
            # print("final seq_x:", seq_x.shape)

            inter_timestep_vector = begin_arr_test[0][end_index]  # vector 64x2 of first timestep of interpolating track
            seq_y = inter_timestep_vector
            seq_y = seq_y.reshape((64 * 2))
            # print("final seq_y:", seq_y.shape)

            X_test.append(seq_x)
            y_test.append(seq_y)
        print("X,y_test_mini", (k + 1) * 100 / (len(begin_arr_train)), "% little songs processed")
    print("X_test_mini", np.asarray(X_test).shape)
    print("y_test_mini",np.asarray(y_test).shape)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    #'''

timesteps = 3
n_features = 64 * 2  # 64x2 note posibilities for begin track + 64x2 for end track = 128x2 (prediction is interp track=64 as output size)

if training:
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(256, activation='relu', input_shape=(timesteps, n_features), return_sequences=True)))
    #model.add(LSTM(256, activation='sigmoid', input_shape=(timesteps, n_features), return_sequences=False))

    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, activation='relu', return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(256, activation='relu')))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    #model.summary()
    model.compile(optimizer=opt, loss='binary_crossentropy')
    #model.summary()

    # SAVE MODEL AND WEIGHTS
    '''
    checkpoint_path = "final_interpolation_dataset_np/lstm_model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    '''
    # fit model
    history = model.fit(X, y, validation_data=(X_test, y_test), epochs=3, batch_size=64)#, callbacks=[cp_callback]) #validation_data=(X_test, y_test), epochs=1)
    model.save("final_interpolation_dataset_np/my_lstm_model")

    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()



if evaluating:

    model = keras.models.load_model("final_interpolation_dataset_np/my_lstm_model")

    # demonstrate prediction
    threshold = 0.5
    predicted_interpolation = []
    x_input = state_test

    print("FIRST PREDICTION")
    print("x_input shape is", x_input.shape) #3x64x2 np
    x_input = x_input.reshape((1, timesteps, n_features))
    print("x_input shape is", x_input.shape)  # 1x3x128 np
    yhat = model.predict(x_input, verbose=0)
    #print(yhat)
    print("yhat", type(yhat), yhat.shape) #1x128 np
    yhat = yhat.reshape((64, 2))
    #print(yhat)
    print(type(yhat), yhat.shape) #64x2 np
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    y_hat_1 = yhat
    print(type(yhat), yhat.shape) #64x2 np
    predicted_interpolation.append(y_hat_1)

    print("SECOND PREDICTION")
    x_input = state_test[1:,:,:]
    print("x_input after cut is", type(x_input), x_input.shape) # 2x64x2
    y_hat_1 = y_hat_1.reshape((1, y_hat_1.shape[0], y_hat_1.shape[1])) # 64x2 --> 1x64x2 in order to concat with x_input
    x_input = np.concatenate((x_input, y_hat_1), axis=0)
    print("x_input shape is", x_input.shape)  # 3x64x2
    x_input = x_input.reshape((1, timesteps, n_features))
    print("x_input shape is", x_input.shape)  # 1x3x128 np
    yhat = model.predict(x_input, verbose=0)
    yhat = yhat.reshape((64, 2))
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    y_hat_2 = yhat
    predicted_interpolation.append(y_hat_2)

    print("THIRD PREDICTION")
    x_input = state_test[2:,:,:]
    y_hat_2 = y_hat_2.reshape((1, y_hat_2.shape[0], y_hat_2.shape[1]))
    x_input = np.concatenate((x_input, y_hat_1), axis=0)
    x_input = np.concatenate((x_input, y_hat_2), axis=0)
    print("x_input shape is", x_input.shape)  # 3x64x2
    x_input = x_input.reshape((1, timesteps, n_features))
    print("x_input shape is", x_input.shape)  # 1x3x128 np
    yhat = model.predict(x_input, verbose=0)
    yhat = yhat.reshape((64, 2))
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    y_hat_3 = yhat
    predicted_interpolation.append(y_hat_3)

    y_hat_3 = y_hat_3.reshape((1, y_hat_3.shape[0], y_hat_3.shape[1]))
    y_hats = np.concatenate((y_hat_1,y_hat_2), axis=0)
    y_hats = np.concatenate((y_hats,y_hat_3), axis=0)
    print("made y_hats shape is", y_hats.shape)  # 3x64x2

    print("FOURTH AND ON PREDICTIONS")
    x_input = y_hats
    print("x_input of y_hats shape is", x_input.shape)  # 3x64x2
    x_input = x_input.reshape((1, timesteps, n_features))
    print("x_input of y_hats shape is", x_input.shape)  # 1x3x128 np
    yhat = model.predict(x_input, verbose=0)
    yhat = yhat.reshape((64, 2))
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    new_yhat = yhat
    predicted_interpolation.append(new_yhat)

    new_yhat = new_yhat.reshape((1, new_yhat.shape[0], new_yhat.shape[1]))
    y_hats = np.concatenate((y_hats,new_yhat), axis=0)
    print("before loop y_hats shape is", y_hats.shape)  # 3x64x2

    for j in range(4,42):
        x_input = y_hats[1:,:,:]
        y_hats = y_hats[1:,:,:]
        #x_input = np.concatenate((x_input,new_yhat), axis=0)
        print("1-------------------------------------------------------------")
        print("loop x_input shape is", x_input.shape)  # 3x64x2
        x_input = x_input.reshape((1, timesteps, n_features))
        print("loop x_input shape is", x_input.shape)  # 1x3x128 np
        print("2-------------------------------------------------------------")
        yhat = model.predict(x_input, verbose=0)
        yhat = yhat.reshape((64, 2))
        yhat[yhat >= threshold] = 1
        yhat[yhat < threshold] = 0
        predicted_interpolation.append(yhat)
        yhat = yhat.reshape((1, yhat.shape[0], yhat.shape[1]))
        #print("loop y_hats shape is", y_hats.shape)  # 1x3x128 np
        y_hats = np.concatenate((y_hats,yhat), axis=0)
        #print("loop y_hats shape is", y_hats.shape)  # 1x3x128 np
        #new_yhat = yhat

    predicted_interpolation = np.asarray(predicted_interpolation)
    print("predicted_interpolation:", predicted_interpolation.shape)
    total_np = np.concatenate((state_1_last, predicted_interpolation), axis=0)
    total_np = np.concatenate((total_np, state_3_last), axis=0)
    print(total_np.shape)
    noteStateMatrixToMidi(total_np, folder_name + '/' + song_name_1 + '_interp_LSTM')  # save as midi