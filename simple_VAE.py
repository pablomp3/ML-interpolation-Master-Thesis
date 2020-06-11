from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import os
import pandas as pd
#import midi
from midi_state_conversion import midiToNoteStateMatrix
from midi_state_conversion import noteStateMatrixToMidi

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import models
import utils
from torchvision.utils import save_image



# pad to add to tracks in order to reach a shape of 43x44x2
pad_44x2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
pad_64x2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
pad_78x2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
mini_pad = [[0,0]]
# --------------
check_gpu = False
fix_data = True
directly_feed_data_in = False
training = False
remove_duplicates = False
target_length = 64 #64 #default = 44
pad = (pad_64x2) #default = pad_44x2
print("pad type:", type(pad), len(pad))
pad = np.asarray(pad)
print("pad type:", type(pad), pad.shape)
# --------------
if check_gpu:
    print(torch.__version__)
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    print(torch.cuda.current_device())
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    print(torch.cuda.is_available())

folder_in = "final_interpolation_dataset" #"songs"
#folder_in = "final_interpolation_dataset_unseen"
folder_in_np = "final_interpolation_dataset_np"
#directory = sorted(os.listdir(folder_in))[:]
directory = os.listdir(folder_in)[50000:]
#random.shuffle(directory)
print(len(directory))

if remove_duplicates:
    for i in range(1,len(directory)+1):
        last_song = directory[i-1]
        current_song = directory[i]
        curr_indicator = current_song[-9:-8]
        last_indicator = last_song[-9:-8]
        if curr_indicator=='_': #single number
            curr_index = current_song[-8:-5]
            curr_first = current_song[-8:-7]
            curr_second = current_song[-6:-5]
        else: #double number
            curr_index = current_song[-9:-5]
            curr_first = current_song[-9:-7]
            curr_second = current_song[-6:-5]

        if last_indicator=='_': #single number
            last_index = last_song[-8:-5]
            last_first = last_song[-8:-7]
            last_second = last_song[-6:-5]
        else: #double number
            last_index = last_song[-9:-5]
            last_first = last_song[-9:-7]
            last_second = last_song[-6:-5]

        #print(last_song)
        #print(current_song)
        #print(curr_index, "// current", curr_first, curr_second, "// previous", last_first, last_second)
        if int(curr_first)==int(last_first)+1:
            os.remove(folder_in + '/' + current_song)
            #print("                delete", current_song)
        print("processed song", i+1, "/", len(directory), ":", (i+1) * 100 / len(directory), "%")

if fix_data:
    print('pad shape', np.shape(pad))
    #remove_rows = []
    #remove_values = []
    padded = []
    delete = []
    correct = []
    #np_correct = []
    #total = [] #total list to add all song matrices
    element = 0
    print('target length:', target_length) #or 44
    directory = directory[:]
    for i in range(0,len(directory)):
        #print(folder_in+"/"+directory[i])
        song = midiToNoteStateMatrix(folder_in+"/"+directory[i])
        #print("song: ", type(song), np.asarray(song).shape, np.asarray(song)[0].shape)

        if np.shape(song)[0] > target_length: # or np.shape(song)[0] < 33: #delete file
            print("file to delete")
            delete.append(directory[i])
        if np.shape(song)[0] < target_length:
            while np.shape(song)[0] < target_length:
                song.append(pad)
            padded.append(directory[i])
        #print("final 0", np.shape(song), type(song))
        #noteStateMatrixToMidi(np.asarray(song), name=('checkpoints/new_track_'+str(i)))
        '''
        if target_length == target_length:
            print("final 1", np.shape(song), type(song))
            #noteStateMatrixToMidi(song, name='checkpoints/new_track_8_1')
            print(np.shape(song)[1])
            while np.shape(song)[1] < target_length:
                song = np.array([np.append(element, mini_pad, axis=0) for element in song])
            print("final 2", np.shape(song), type(song))
            #print(song)
            #noteStateMatrixToMidi(song, name='checkpoints/new_track_8_1')
        '''
        song = np.asarray(song)
        #print("song: ", type(song), song.shape, song[0].shape)
        #noteStateMatrixToMidi(song, name= folder_out + '/' + directory[i] + '_pad_64')
        print("saved song", i, "/", len(directory), ":", i * 100 / len(directory), "%")
        #print("song after: ", type(song), song.shape)
        #np_correct.append(song)
        correct.append(song)

    print("songs to delete:", len(delete))
    print("songs that have been fixed:", len(padded))
    print("songs correct:", len(correct), type(correct))

    correct = np.asarray(correct) #prepare data as np to feed the VAE
    #np_correct = np.asarray(np_correct)
    print("songs correct prepared to feed in:", len(correct), type(correct), correct.shape)
    #print("songs correct prepared to feed in:", len(np_correct), type(np_correct), np_correct.shape)


    # change order of dimensions: (examples, 64, 64, 2) to (examples, 2, 64, 64)
    feed_in = np.einsum('iklj->ijkl', correct)
    print("feed in: ", feed_in.shape)
    feed_in = feed_in.astype(np.float32) # set to float in order to keep data consistency
    print("feed in as float")

if directly_feed_data_in:
    directory = directory[:]
    for i in range(0, len(directory)):
        print(folder_in + "/" + directory[i])
        song = midiToNoteStateMatrix(folder_in + "/" + directory[i])
        print("song: ", type(song), np.asarray(song).shape, np.asarray(song)[0].shape)

# MNIST dataset
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train", type(x_train), x_train.shape, "y_train", type(y_train), y_train.shape, "x_test", type(x_test), x_test.shape, "y_test", type(y_test), y_test.shape)
image_size = x_train.shape[1] #28 #change by
original_dim = image_size * image_size #28x28 #change by 43x44x2
original_dim_new = 44*44*2
'''

'''feed_in = np.random.shuffle(feed_in)
percent_split_80 = int(len(directory)*0.8)
print("split:", percent_split_80)'''

np.save(folder_in_np + '/data[50000:].npy', feed_in) # save
#feed_in = np.load(folder_in_np + '/data.npy') # load
print(type(feed_in), feed_in.shape)
print("AWESOME DATA")
#feed_in_2 = np.load(folder_in_np + '/data2.npy') # load
#print(type(feed_in_2), feed_in_2.shape)
print("AWESOME DATA 2")
#new_feed = np.concatenate((feed_in, feed_in_2), axis=0)
#print(type(new_feed), new_feed.shape)

#np.random.shuffle(new_feed)
#x_train_new, x_test_new = train_test_split(new_feed, test_size=0.05, random_state=42)
'''
percent_split_80 = int(len(directory)*0.8)
print("split:", percent_split_80)

x_train_new = feed_in[:percent_split_80]
print("x_train_new", x_train_new.shape)
x_test_new = feed_in[percent_split_80:]
print("x_test_new", x_test_new.shape)
'''
print("x_train_new", x_train_new.shape)
print("x_test_new", x_test_new.shape)
'''
x_train = np.reshape(x_train, [-1, original_dim])
x_train_new = np.reshape(x_train_new, [-1, original_dim_new])
print("x_train", x_train.shape, type(x_train))
print("x_train_new", x_train_new.shape, type(x_train_new))
#print(x_train[0])
x_test = np.reshape(x_test, [-1, original_dim])
x_test_new = np.reshape(x_test_new, [-1, original_dim_new])
print("x_test", x_test.shape)
print("x_test_new", x_test_new.shape)
#print(x_train[0])
x_train = x_train.astype('float32') / 255 #normalization
x_test = x_test.astype('float32') / 255

x_train = x_train_new
x_test = x_test_new
'''

'''
#x_train_new = np.ones((200,3,64,64))
#x_train_new = np.ones((200,2,64,64))
x_train_new = feed_in
print("x_train_new test w/ np.ones", x_train_new.shape)
#x_train_new = np.ones((200,2,44,44))
x_train_new = x_train_new.astype(np.float32)
print("x_train_new test w/ np.ones", x_train_new.shape)
'''
#-----------------------------------------------------------------------------------------------------------------------
# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.
#-------------------------------------------- models finish ------------------------------------------------------------

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss

if training:
    # parameters
    BATCH_SIZE = 256
    TEST_BATCH_SIZE = 10
    EPOCHS = 10000 #400

    LATENT_SIZE = 100
    LEARNING_RATE = 1e-3

    USE_CUDA = True
    PRINT_INTERVAL = 100
    LOG_PATH = './logs/log.pkl'
    MODEL_PATH = './checkpoints/'
    COMPARE_PATH = './comparisons/'

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    print(torch.cuda.is_available())

    # training code
    train_ids, test_ids = prep.split_dataset()
    print('num train_images:', len(train_ids))
    print('num test_images:', len(test_ids))

    data_train = prep.ImageDiskLoader(train_ids)
    data_test = prep.ImageDiskLoader(test_ids)
    print(data_train)

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    #train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(x_train_new, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    #test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(x_test_new, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    print('latent size:', LATENT_SIZE)

    #model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
    model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if __name__ == "__main__":

        start_epoch = model.load_last_model(MODEL_PATH) + 1
        train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))

        for epoch in range(start_epoch, EPOCHS + 1):
            train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
            test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

            #save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            utils.write_log(LOG_PATH, (train_losses, test_losses))

            model.save_model(MODEL_PATH + '%03d.pt' % epoch)

    #-----------------------------------------------------------------------------------------------------------------------

'''
# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 10 #50
#-----------------------------------------
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

if training:
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        help_ = "Load h5 model trained weights"
        parser.add_argument("-w", "--weights", help=help_)
        help_ = "Use mse loss instead of binary cross entropy (default)"
        parser.add_argument("-m",
                            "--mse",
                            help=help_, action='store_true')
        args = parser.parse_args()
        models = (encoder, decoder)
        data = (x_test, y_test)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs,
                                                      outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
        plot_model(vae,
                   to_file='vae_mlp.png',
                   show_shapes=True)

        if args.weights:
            vae.load_weights(args.weights)
        else:
            # train the autoencoder
            vae.fit(x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, None))
            #vae.save_weights('vae_mlp_mnist.h5')

        plot_results(models,
                     data,
                     batch_size=batch_size,
                     model_name="vae_mlp")
        print("the job is done")
        
'''