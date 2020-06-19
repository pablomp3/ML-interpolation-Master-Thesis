import tensorflow.compat.v1 as tf
import numpy as np
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
load_data = True
build_graph = True
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

if build_data:
    folder_name = 'final_interpolation_dataset_flow_test'
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
    np.save('final_interpolation_dataset_np' + '/encoded_z_begin.npy', begin_arr) # save
    np.save('final_interpolation_dataset_np' + '/encoded_z_interp.npy', inter_arr) # save
    np.save('final_interpolation_dataset_np' + '/encoded_z_end.npy', end_arr) # save

if load_data:
    begin_arr = np.load('final_interpolation_dataset_np' + '/encoded_z_begin.npy')  # load
    print(type(begin_arr), begin_arr.shape)
    inter_arr = np.load('final_interpolation_dataset_np' + '/encoded_z_interp.npy')  # load
    print(type(inter_arr), inter_arr.shape)
    end_arr = np.load('final_interpolation_dataset_np' + '/encoded_z_end.npy')  # load
    print(type(end_arr), end_arr.shape)

    limit = int(0.8 * len(begin_arr))
    begin_arr_train = begin_arr[:limit]
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
    learning_rate = 0.5
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
        h1 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1")
        h2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h2")

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
        z_pred = tf.nn.softmax(tf.add(tf.linalg.matmul(z1, h1), tf.linalg.matmul(z2, h2))) #z_pred = softmax(z1*H1 + z2*H2)

        loss = tf.losses.mean_squared_error(z, z_pred)

        optimizer = tf.train.AdamOptimizer()
        # add an optimiser
        #optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

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
        session.run(init_op)
        for i in range(0, 1000):
            loss_value, _ = session.run([loss, train_op], {z1: begin_arr_train, z2: end_arr_train, z: inter_arr_train}) #{z1: z1_value, z2: z2_value, z: z_value})
            print(loss_value)
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