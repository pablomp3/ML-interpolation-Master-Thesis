import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import analyze
import models
import tensorflow
from midi_state_conversion import midiToNoteStateMatrix
from midi_state_conversion import noteStateMatrixToMidi
from midi_state_conversion import padStateMatrix
pad_64x2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
USE_CUDA = True
#MODEL = 'dfc-300'
MODEL = 'pablo_full_data_4000'
#MODEL = 'pablo_full_200'
IMAGE_PATH = 'checkpoints/'
MODEL_PATH = './checkpoints/' + MODEL
LOG_PATH = './logs/' + MODEL + '/log.pkl'
OUTPUT_PATH = './samples/'
PLOT_PATH = './plots/' + MODEL
LATENT_SIZE = 100

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
#model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)
print('latent size:', model.latent_size)
#attr_map, id_attr_map = prep.get_attributes()

#model.load_model(file_path=test_path)
model.load_last_model(MODEL_PATH)

# ------------------------------------------
# ------------------------------------------
build_data = False
load_data = False
build_graph = False
restore = True
# ------------------------------------------
# ------------------------------------------

def shuffle_in_unison_scary(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    #return a, b, c

# COPYPASTE 'final_interpolation_dataset_full' (~90.000 tracks formato 1_2_3) to 'final_interpolation_dataset_full_encoded'
# if directory[i] de 'final_interpolation_dataset_unseen' contiene song_x_1, eliminar en 'final_interpolation_dataset_full_encoded' -->
# eliminar song_x_1, x_2, x_3
# song_x-1_3
# hay que tener cuidado con los ejemplos de song_10_2, 10_3, 11_2 // song_10_3, 11_2, 11_3 (hay que eliminar tracks completas de antes y despues)

if build_data:
    #folder_name = 'final_interpolation_dataset_flow_test'
    folder_name = 'final_interpolation_dataset_full_encoded'
    directory = sorted(os.listdir(folder_name))[:]
    begin_arr = []
    begin_arr_name = []
    inter_arr = []
    inter_arr_name = []
    end_arr = []
    end_arr_name = []
    other = []
    feed_in = []
    print("total len:", len(directory))
    for i in range(0,len(directory)):
        song_name = directory[i]
        track_type = song_name[-6:-5]
        if track_type=='1':
            begin_arr_name.append(song_name)
        if track_type=='2':
            inter_arr_name.append(song_name)
        if track_type=='3':
            end_arr_name.append(song_name)
        target_length = 64
        pad = pad_64x2
        # ---------------------------------------------------------#
        # target_length #  pad   # lowerBound # upperBound # span #
        #     44         pad_44x2      36           80        44  #
        #     64         pad_64x2      28           92        64  #
        #     78         pad_78x2      22           100       78  #
        # ---------------------------------------------------------#
        state = midiToNoteStateMatrix(folder_name + '/' + song_name)  # shape 43x64x2
        state = padStateMatrix(folder_name, song_name, target_length, pad, save_as_midi=False)  # shape 64x64x2

        state = np.einsum('ijk->kij', state)  # shape 2x64x64
        state = state.astype(np.float32)  # set to float in order to keep data consistency
        #print("reshaped to:", type(state), state.shape)

        state = torch.from_numpy(state)  # convert state: numpy array to torch tensor
        #print("song to encode:", type(state), state.shape)
        encoded_song = analyze.get_z(state, model, device)
        #print("encoded song z:", type(encoded_song), encoded_song.shape)
        encoded_song = encoded_song.cpu().numpy()
        #print("encoded song z:", type(encoded_song), encoded_song.shape)
        if track_type=='1':
            begin_arr.append(encoded_song)
        if track_type=='2':
            inter_arr.append(encoded_song)
        if track_type=='3':
            end_arr.append(encoded_song)
        #feed_in.append(encoded_song)
        print("processed song", i + 1, "/", len(directory), ":", (i + 1) * 100 / len(directory), "%")
        #print("")

    print("begin:", len(begin_arr))
    print("interp:", len(inter_arr))
    print("end:", len(end_arr))
    print("error:", len(other))

    #feed_in = np.asarray(feed_in)  # prepare encoded data as a np array of np arrays [examples, 1, 100] to feed the NN
    #feed_in = feed_in.astype(np.float32)  # set to float in order to keep data consistency
    begin_arr = np.asarray(begin_arr)
    inter_arr = np.asarray(inter_arr)
    end_arr = np.asarray(end_arr)

    begin_arr = begin_arr.astype(np.float32)
    inter_arr = inter_arr.astype(np.float32)
    end_arr = end_arr.astype(np.float32)

    shuffle_in_unison_scary(begin_arr, inter_arr, end_arr) #shuffle equally the 3 arrays
    shuffle_in_unison_scary(begin_arr_name, inter_arr_name, end_arr_name)

    print("begin z1 songs correct prepared to feed in:", type(begin_arr), begin_arr.shape)
    #print(begin_arr)
    #print(begin_arr_name) #print to check if the shuffle has been made equally in the 3 arrays
    print("interp z songs correct prepared to feed in:", type(inter_arr), inter_arr.shape)
    #print(inter_arr)
    #print(inter_arr_name)
    print("end z2 songs correct prepared to feed in:", type(end_arr), end_arr.shape)
    #print(end_arr)
    #print(end_arr_name)
    np.save('final_interpolation_dataset_np' + '/encoded_full_z_begin.npy', begin_arr) # save
    np.save('final_interpolation_dataset_np' + '/encoded_full_z_interp.npy', inter_arr) # save
    np.save('final_interpolation_dataset_np' + '/encoded_full_z_end.npy', end_arr) # save

if load_data:
    begin_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_begin.npy')  # load
    print(type(begin_arr), begin_arr.shape)
    inter_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_interp.npy')  # load
    print(type(inter_arr), inter_arr.shape)
    end_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_end.npy')  # load
    print(type(end_arr), end_arr.shape)

    limit = int(0.8 * len(begin_arr))
    begin_arr_train = begin_arr[:limit]
    print("BEGIN ARR TRAIN:", begin_arr_train.dtype, begin_arr_train.shape)
    begin_arr_test = begin_arr[limit:]
    inter_arr_train = inter_arr[:limit]
    inter_arr_test = inter_arr[limit:]
    end_arr_train = end_arr[:limit]
    end_arr_test = end_arr[limit:]

if build_graph:
    '''graph = tf.Graph()
    #session = tf.Session(graph=graph)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)'''

    # Python optimisation variables
    learning_rate = 0.0001
    epochs = 10
    batch_size = 100

    '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print("mnist:", type(mnist))
    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    print("mnist:", type(batch_x), batch_x.shape) #numpy array (100, 784)
    '''

    #with graph.as_default():
    with tf.Session() as session:

        #declare the training data placeholders
        # input z1 = begining encoded track (1,100)
        z1 = tf.placeholder(tf.float32, [None, 1, 100], name='z1')
        # input z2 = end encoded track (1,100)
        z2 = tf.placeholder(tf.float32, [None, 1, 100], name='z2')
        # output z = interpolation encoded track (1,100)
        z = tf.placeholder(tf.float32, [None, 1, 100], name='z')

        # now declare the weights connecting the input to the hidden layer to output
        h1_1 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_1")
        h1_2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_2")
        b1 = tf.Variable(tf.random_normal([100]), name='b1')

        h2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h2")
        b2 = tf.Variable(tf.random_normal([100]), name='b2')

        h3 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h3")
        b3 = tf.Variable(tf.random_normal([100]), name='b3')

        h4 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h4")
        b4 = tf.Variable(tf.random_normal([100]), name='b4')

        '''
        W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([300]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([10]), name='b2')
        
        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(x, W1), b1)    # z = Wx + b
        hidden_out = tf.nn.relu(hidden_out)          # h = f(z)
        
        # now calculate the hidden layer output - in this case, let's use a softmax activated
        # output layer
        y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
        '''

        # output layer
        #z_pred = tf.linalg.matmul(z1,h1) + tf.linalg.matmul(z2,h2)
        hidden_sum = tf.add(tf.matmul(z1, h1_1), tf.matmul(z2, h1_2)) # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        hidden_mid_1 = tf.add(hidden_sum, b1) # hidden_out = (z1*h1 + z2*h2) + b1 == [1, 100]
        hidden_mid_1 = tf.nn.tanh(hidden_mid_1) # [1, 100]

        hidden_mid_2 = tf.add(tf.matmul(hidden_mid_1, h2), b2)
        hidden_mid_2 = tf.nn.tanh(hidden_mid_2)

        hidden_mid_3 = tf.add(tf.matmul(hidden_mid_2, h3), b3)
        hidden_mid_3 = tf.nn.tanh(hidden_mid_3)

        hidden_final = tf.add(tf.matmul(hidden_mid_3, h4), b4) #, tf.matmul(z2, h2_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        #hidden_out_2 = tf.matmul(hidden_out, h2_1) #, tf.matmul(z2, h2_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        z_pred = tf.nn.leaky_relu(hidden_final) #0.91
        y_ = z_pred
        y = z

        loss = tf.losses.mean_squared_error(labels = z, predictions = z_pred)
        #loss = tf.losses.mean_pairwise_squared_error(labels = z, predictions = z_pred)
        '''
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.reshape(ylogits, [-1]),
                labels=tf.cast(labels, dtype=tf.float32)))
                '''
        #loss = tf.losses.sigmoid_cross_entropy(z, z_pred) #0.28 loss

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients)

        # finally setup the initialisation operator
        #init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #z1_value = np.random.rand(1,100)
        #z1_value = begin_arr
        #z2_value = np.random.rand(1,100)
        #z2_value = end_arr
        #z_value = np.ones((1,100))
        #z_value = inter_arr

        '''
        init_op = tf.initialize_all_variables()
        session.run(init_op)
        for i in range(0,20):
            loss_value, _ = session.run([loss, train_op], {z1: z1_value, z2:z2_value, z:z_value})
            print(loss_value)
        '''
        '''
        init_op = tf.initialize_all_variables()
        session.run(init_op)
        total_batch = int(len(begin_arr) / batch_size)
        for epoch in range(20):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size) #numpy array (100, 784)
                _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    '''
        init_op = tf.initialize_all_variables()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        session.run(init_op)

        train_x = []
        train_l = []
        test_x = []
        test_l = []
        for i in range(0, 2000):
            loss_value_train, _ = session.run([loss, train_op], {z1: begin_arr_train, z2: end_arr_train, z: inter_arr_train}) #{z1: z1_value, z2: z2_value, z: z_value})
            loss_value_test, _ = session.run([loss, train_op], {z1: begin_arr_test, z2: end_arr_test,  z: inter_arr_test})  # {z1: z1_value, z2: z2_value, z: z_value})

            train_x.append(i)
            test_x.append(i)
            train_l.append(loss_value_train)
            test_l.append(loss_value_test)
            if i%50==0:
                print("iter", i + 1, "| train loss:", loss_value_train, "| test loss:", loss_value_test)
            #print(h2_1.eval())
            #print(z_pred.dtype)

        folder_name = 'final_interpolation_dataset_unseen'
        song_name_1 = '4d304a5af6078632e2ea610a22c1f84d_2727.midi_5_1.midi'
        interp_real = '4d304a5af6078632e2ea610a22c1f84d_2727.midi_5_2.midi'
        song_name_2 = '4d304a5af6078632e2ea610a22c1f84d_2727.midi_5_3.midi'
        target_length = 64
        pad = pad_64x2
        # ---------------------------------------------------------#
        # target_length #  pad   # lowerBound # upperBound # span #
        #     44         pad_44x2      36           80        44  #
        #     64         pad_64x2      28           92        64  #
        #     78         pad_78x2      22           100       78  #
        # ---------------------------------------------------------#
        state_1 = midiToNoteStateMatrix(folder_name + '/' + song_name_1)  # shape 43x64x2
        state_2 = midiToNoteStateMatrix(folder_name + '/' + song_name_2)  # shape 43x64x2
        state_3 = midiToNoteStateMatrix(folder_name + '/' + interp_real)  # shape 43x64x2
        state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_1_last = state_1[0:42]
        state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_2_last = state_2[0:42]
        state_3 = padStateMatrix(folder_name, interp_real, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_3_last = state_3[0:42]

        # print(state_2[40])
        state_1 = np.einsum('ijk->kij', state_1)  # shape 2x64x64
        state_2 = np.einsum('ijk->kij', state_2)  # shape 2x64x64
        state_3 = np.einsum('ijk->kij', state_3)  # shape 2x64x64
        state_1 = state_1.astype(np.float32)  # set to float in order to keep data consistency
        state_2 = state_2.astype(np.float32)  # set to float in order to keep data consistency
        state_3 = state_3.astype(np.float32)  # set to float in order to keep data consistency
        state_1 = torch.from_numpy(state_1)  # convert state: numpy array to torch tensor
        state_2 = torch.from_numpy(state_2)  # convert state: numpy array to torch tensor
        state_3 = torch.from_numpy(state_3)  # convert state: numpy array to torch tensor

        model.eval()
        z1_example = analyze.get_z(state_1, model, device)
        #print("z1_example:", z1_example, z1_example.shape, type(z1_example))
        z1_example = z1_example.cpu().numpy()
        z1_example = np.expand_dims(z1_example, axis=0)
        print("z1_example:", z1_example, z1_example.shape, type(z1_example))
        z2_example = analyze.get_z(state_2, model, device)
        z2_example = z2_example.cpu().numpy()
        z2_example = np.expand_dims(z2_example, axis=0)
        #print("z2_example:", z2_example, z2_example.shape)
        z3_example = analyze.get_z(state_3, model, device)
        z3_example = z3_example.cpu().numpy()
        z3_example = np.expand_dims(z3_example, axis=0)

        feed_dict = {z1: z1_example, z2: z2_example}
        interp_encoded = session.run(z_pred, feed_dict)
        print("PREDICTION:")
        print(interp_encoded) #np array (1,1,100) --> to class torch ([1, 100])
        interp_encoded = np.squeeze(interp_encoded, axis=0) #np array (1,100)
        print("features:", type(interp_encoded), interp_encoded.shape)
        interp_encoded = torch.from_numpy(interp_encoded).to(device) #torch tensor
        print("features:", type(interp_encoded), interp_encoded.shape)
        print("REAL INTERP Z")
        print(z3_example)

        result = []
        model.eval()
        with torch.no_grad():
            im = torch.squeeze(model.decode(interp_encoded).cpu())
            result.append(im)
        k = 0
        for t in result:
            print(k)
            print(t.numpy().shape, type(t.numpy()))  # numpy array of size (3, 64, 64) each
            nump_song = t.numpy()
            # print("reshaped to:", type(nump_song), nump_song.shape)
            nump_song = np.einsum('kij->ijk', nump_song)  # shape 64x64x2 (as original)
            print("reshaped to:", type(nump_song), nump_song.shape)
            nump_song = nump_song[0:42]
            nump_song[nump_song >= .2] = 1
            nump_song[nump_song < .2] = 0
            # print(nump_song[40])
            if k == 0:
                total_np = nump_song
            else:
                total_np = np.concatenate((total_np, nump_song), axis=0)
            print(total_np.shape)

            k += 1
        print("last:", total_np.shape)
        total_np = np.concatenate((state_1_last, total_np), axis=0)
        total_np = np.concatenate((total_np, state_2_last), axis=0)
        # print(total_np[30])
        noteStateMatrixToMidi(total_np, 'flow_songs/flow' + song_name_1 + '_interp_' + str(k))  # save as midi

        save_path = saver.save(session, 'flow_songs/ANN_model_' + str(i+1) + '.ckpt')
        print("Model saved in path: %s" % save_path)
        plt.figure()
        plt.title('Train Loss vs. Test Loss')
        plt.xlabel('episodes')
        plt.ylabel('loss')
        plt.plot(train_x, train_l, 'b', label='train_loss')
        plt.plot(test_x, test_l, 'r', label='test_loss')
        plt.legend()
        plt.savefig('flow_songs/loss_' + str(i+1) + '.png')
        plt.show()
        '''
        # start the session
        with tf.Session() as sess:
           # initialise the variables
           sess.run(init_op)
           total_batch = int(len(mnist.train.labels) / batch_size)
           for epoch in range(epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                     _, c = sess.run([optimiser, cross_entropy], 
                                 feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
           print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        '''
if restore:
    # with graph.as_default():
    tf.reset_default_graph()
    with tf.Session() as session:

        # declare the training data placeholders
        # input z1 = begining encoded track (1,100)
        z1 = tf.placeholder(tf.float32, [None, 1, 100], name='z1')
        # input z2 = end encoded track (1,100)
        z2 = tf.placeholder(tf.float32, [None, 1, 100], name='z2')
        # output z = interpolation encoded track (1,100)
        z = tf.placeholder(tf.float32, [None, 1, 100], name='z')

        # now declare the weights connecting the input to the hidden layer to output
        h1_1 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_1")
        h1_2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_2")
        b1 = tf.Variable(tf.random_normal([100]), name='b1')

        h2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h2")
        b2 = tf.Variable(tf.random_normal([100]), name='b2')

        h3 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h3")
        b3 = tf.Variable(tf.random_normal([100]), name='b3')

        h4 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h4")
        b4 = tf.Variable(tf.random_normal([100]), name='b4')

        # output layer
        # z_pred = tf.linalg.matmul(z1,h1) + tf.linalg.matmul(z2,h2)
        hidden_sum = tf.add(tf.matmul(z1, h1_1), tf.matmul(z2, h1_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        hidden_mid_1 = tf.add(hidden_sum, b1)  # hidden_out = (z1*h1 + z2*h2) + b1 == [1, 100]
        hidden_mid_1 = tf.nn.tanh(hidden_mid_1)  # [1, 100]

        hidden_mid_2 = tf.add(tf.matmul(hidden_mid_1, h2), b2)
        hidden_mid_2 = tf.nn.tanh(hidden_mid_2)

        hidden_mid_3 = tf.add(tf.matmul(hidden_mid_2, h3), b3)
        hidden_mid_3 = tf.nn.tanh(hidden_mid_3)

        hidden_final = tf.add(tf.matmul(hidden_mid_3, h4), b4)  # , tf.matmul(z2, h2_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        # hidden_out_2 = tf.matmul(hidden_out, h2_1) #, tf.matmul(z2, h2_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
        z_pred = tf.nn.leaky_relu(hidden_final)  # 0.91

        '''
        loss = tf.losses.mean_squared_error(labels=z, predictions=z_pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients)

        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()

        session.run(init_op)'''

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Restore variables from disk.
        saver.restore(session, 'flow_songs/ANN_model_2000.ckpt')
        print("Model restored.")

        folder_name = 'final_interpolation_dataset_unseen'
        song_name_1 = '947aecbf5e48d5ee7282abe1b815bb86_2173.midi_18_1.midi'
        interp_real = '947aecbf5e48d5ee7282abe1b815bb86_2173.midi_18_2.midi'
        song_name_2 = '947aecbf5e48d5ee7282abe1b815bb86_2173.midi_18_3.midi'
        target_length = 64
        pad = pad_64x2
        # ---------------------------------------------------------#
        # target_length #  pad   # lowerBound # upperBound # span #
        #     44         pad_44x2      36           80        44  #
        #     64         pad_64x2      28           92        64  #
        #     78         pad_78x2      22           100       78  #
        # ---------------------------------------------------------#
        state_1 = midiToNoteStateMatrix(folder_name + '/' + song_name_1)  # shape 43x64x2
        state_2 = midiToNoteStateMatrix(folder_name + '/' + song_name_2)  # shape 43x64x2
        state_3 = midiToNoteStateMatrix(folder_name + '/' + interp_real)  # shape 43x64x2
        state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_1_last = state_1[0:42]
        state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_2_last = state_2[0:42]
        state_3 = padStateMatrix(folder_name, interp_real, target_length, pad, save_as_midi=False)  # shape 64x64x2
        state_3_last = state_3[0:42]

        # print(state_2[40])
        state_1 = np.einsum('ijk->kij', state_1)  # shape 2x64x64
        state_2 = np.einsum('ijk->kij', state_2)  # shape 2x64x64
        state_3 = np.einsum('ijk->kij', state_3)  # shape 2x64x64
        state_1 = state_1.astype(np.float32)  # set to float in order to keep data consistency
        state_2 = state_2.astype(np.float32)  # set to float in order to keep data consistency
        state_3 = state_3.astype(np.float32)  # set to float in order to keep data consistency
        state_1 = torch.from_numpy(state_1)  # convert state: numpy array to torch tensor
        state_2 = torch.from_numpy(state_2)  # convert state: numpy array to torch tensor
        state_3 = torch.from_numpy(state_3)  # convert state: numpy array to torch tensor

        model.eval()
        z1_example = analyze.get_z(state_1, model, device)
        # print("z1_example:", z1_example, z1_example.shape, type(z1_example))
        z1_example = z1_example.cpu().numpy()
        z1_example = np.expand_dims(z1_example, axis=0)
        print("z1_example:", z1_example, z1_example.shape, type(z1_example))
        z2_example = analyze.get_z(state_2, model, device)
        z2_example = z2_example.cpu().numpy()
        z2_example = np.expand_dims(z2_example, axis=0)
        # print("z2_example:", z2_example, z2_example.shape)
        z3_example = analyze.get_z(state_3, model, device)
        z3_example = z3_example.cpu().numpy()
        z3_example = np.expand_dims(z3_example, axis=0)

        feed_dict = {z1: z1_example, z2: z2_example}
        interp_encoded = session.run(z_pred, feed_dict)
        print("PREDICTION:")
        print(interp_encoded)  # np array (1,1,100) --> to class torch ([1, 100])
        interp_encoded = np.squeeze(interp_encoded, axis=0)  # np array (1,100)
        print("features:", type(interp_encoded), interp_encoded.shape)
        interp_encoded = torch.from_numpy(interp_encoded).to(device)  # torch tensor
        print("features:", type(interp_encoded), interp_encoded.shape)
        print("REAL INTERP Z")
        print(z3_example)

        result = []
        model.eval()
        with torch.no_grad():
            # im = torch.squeeze(model.decode(interp_encoded).cpu())
            im = torch.squeeze(model.decode(interp_encoded).cpu())
            result.append(im)
        k = 0
        for t in result:
            print(k)
            print(t.numpy().shape, type(t.numpy()))  # numpy array of size (3, 64, 64) each
            nump_song = t.numpy()
            # print("reshaped to:", type(nump_song), nump_song.shape)
            nump_song = np.einsum('kij->ijk', nump_song)  # shape 64x64x2 (as original)
            print("reshaped to:", type(nump_song), nump_song.shape)
            nump_song = nump_song[0:42]
            nump_song[nump_song >= .15] = 1
            nump_song[nump_song < .15] = 0
            # print(nump_song[40])
            if k == 0:
                total_np = nump_song
            else:
                total_np = np.concatenate((total_np, nump_song), axis=0)
            print(total_np.shape)

            k += 1
        print("last:", total_np.shape)
        total_np = np.concatenate((state_1_last, total_np), axis=0)
        total_np = np.concatenate((total_np, state_2_last), axis=0)
        # print(total_np[30])
        noteStateMatrixToMidi(total_np, 'flow_songs/' + song_name_1 + '_interp_flow' + str(k))  # save as midi