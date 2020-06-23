import torch
import analyze
import models
from torchvision.utils import save_image
import utils
from torchvision import transforms
from torchvision.transforms.functional import crop
import preprocess as prep
import numpy as np
import numpy
import matplotlib.pyplot as plt
import midi
from PIL import Image
from midi_state_conversion import midiToNoteStateMatrix
from midi_state_conversion import noteStateMatrixToMidi
from midi_state_conversion import padStateMatrix
import midi

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

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])
def get_ims(im_ids): #['00001.jpg']
    ims = []
    for im_id in im_ids:
        im_path = IMAGE_PATH + im_id
        im = Image.open(im_path)
        print(im)
        im = crop(im, 30, 0, 178, 178)
        ims.append(im_transform(im))
    return ims

# --------------------------------------------------------
# ---------------------- M O D E -------------------------
# --------------------------------------------------------
generate = False
load_midi_and_convert = False
encoding_decoding_photo = False
encoding_decoding_song = False
interpolating_photo = False
interpolating_song = False
interpolating_flow = False
plot_loss_plot = False
evaluation = True
# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

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

if generate:
    new_song = analyze.generate(model, 1, device)
    save_image(new_song, 'checkpoints/new_song.png', padding=0, nrow=10)

    print("generated by model:", type(new_song), new_song.shape)
    nump_song = np.asarray(new_song)
    print("generated by model as np:", type(nump_song), nump_song.shape)

    # change order of dimensions: (examples, 2, 64, 64) to (examples, 64, 64, 2)
    nump_song = np.einsum('ijkl->iklj', nump_song)
    print("reshaped: ", nump_song.shape)
    #print(nump_song)
    #nump_song = np.squeeze(nump_song)
    nump_song = nump_song[:,:,:,:2]
    print("squeezed:", nump_song.shape)

if load_midi_and_convert:
    from mido import MidiFile

    # ----------------- print notes of song 1 and save it --------------------------------
    song_name = '0a1b7f59058eb2fba0a5bf43295c638d_11328.midi_8_1.midi'
    mid = MidiFile('checkpoints/'+song_name)
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)
    state = midiToNoteStateMatrix('checkpoints/'+song_name) #43x64x2
    state = np.asarray(state)
    print("state of midi song: ", type(state), state.shape)
    #print(state)

    #state = nump_song
    #print("state of generated song: ", type(state), state.shape)
    #state = state[21:, :, 1:]
    print("state of squeezed generated song: ", state.shape, " and save as new track")

    # ----------------- load saved song and print notes of reconstructed song --------------------------------
    noteStateMatrixToMidi(state, name='checkpoints/new_track_110_2')
    mid = MidiFile('checkpoints/new_track_110_2.mid')
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)
    '''
    print("Read new track")
    song_name = 'checkpoints/new_track.mid'
    state = midiToNoteStateMatrix(song_name)
    state = np.asarray(state)
    print("state of new track: ", type(state), state.shape) #43x64x2 (360 bytes)
    '''

if encoding_decoding_photo:
    #new_song = analyze.generate(model, 1, device) #generate image sampled from latent space
    #print("model generated photo:", type(new_song), new_song.shape)
    #print(new_song)
    #print(new_song.type())
    #save_image(new_song, 'checkpoints/yes_song.png', padding=0, nrow=10)

    ''' input image to be encoded and decoded is
        torch tensor of [3, 64, 64]'''
    photo = ['000001.jpg']
    photo = get_ims(photo)
    print("input specs to encoder_decoder:", type(photo[0]), photo[0].shape) #class torch.tensor, size [3,64,64]
    print(photo[0])
    encoded_photo = analyze.get_z(photo[0], model, device)
    print("encoded photo z:", type(encoded_photo), encoded_photo.shape) #class torch.tensor, size [1,100]
    #print(encoded_photo)
    #print(encoded_photo.type())
    model.eval()
    with torch.no_grad():
        decoded_photo = model.decode(encoded_photo).cpu()
    print("decoded photo:", type(decoded_photo), decoded_photo.shape) #class torch.tensor, size [1,3,64,64]
    #print(decoded_photo)
    #print(decoded_photo.type())
    save_image(decoded_photo, 'checkpoints/decoded_song.png', padding=0, nrow=10)

if encoding_decoding_song:
    '''input song as numpy array [43, 64, 2] is padded to
    shape [64, 64, 2] and reconstructed back to midi'''
    prove_reshape_to_midi = False

    folder_name = 'checkpoints/test'
    song_name = 'ff61238332977860aaa35023ca5e0732_9944.midi_4_1.midi'

    target_length = 64
    pad = pad_64x2
    #---------------------------------------------------------#
    # target_length #  pad   # lowerBound # upperBound # span #
    #     44         pad_44x2      36           80        44  #
    #     64         pad_64x2      28           92        64  #
    #     78         pad_78x2      22           100       78  #
    #---------------------------------------------------------#
    state = midiToNoteStateMatrix(folder_name + '/' + song_name)  # shape 43x64x2
    state = padStateMatrix(folder_name, song_name, target_length, pad, save_as_midi=False) # shape 64x64x2
    print(state[10])

    '''prove that is possible reshape [64, 64, 2] to [2, 64, 64] for feeding the NN
    and then reshaping to [64, 64, 2] in order to convert the tensor back to midi'''
    state = np.einsum('ijk->kij', state) # shape 2x64x64
    state = state.astype(np.float32)  # set to float in order to keep data consistency
    print("reshaped to:", type(state), state.shape)
    #print(state)

    state = torch.from_numpy(state)  # convert state: numpy array to torch tensor
    print("numpy to torch tensor:")
    #print(state)
    print("song to encode:", type(state), state.shape)
    encoded_song = analyze.get_z(state, model, device)
    #print(state)
    print("encoded song z:", type(encoded_song), encoded_song.shape)
    #print(encoded_song)

    model.eval()
    with torch.no_grad():
        decoded_song = model.decode(encoded_song).cpu()
    print("decoded song:", type(decoded_song), decoded_song.shape)
    #print(decoded_song)


    nump_song = decoded_song.cpu().numpy()
    print("reshaped to:", type(nump_song), nump_song.shape)
    #print(nump_song)
    nump_song = np.squeeze(nump_song)
    #print(nump_song)
    print("squeezed to:", type(nump_song), nump_song.shape)
    nump_song = np.einsum('kij->ijk', nump_song) # shape 64x64x2 (as original)
    print("reshaped to:", type(nump_song), nump_song.shape)
    print(nump_song[10])
    noteStateMatrixToMidi(nump_song, folder_name + '/' + 'reshaped_3' + song_name) # save as midi

if interpolating_photo:
    ''' input image to be encoded and decoded is
            torch tensor of [3, 64, 64]'''
    photo = ['000001.jpg', '000002.jpg']
    photo = get_ims(photo)
    print("input specs to encoder_decoder:", type(photo[0]), photo[0].shape)  # class torch.tensor, size [3,64,64]
    print(photo[0])
    print("input specs to encoder_decoder:", type(photo[1]), photo[1].shape)  # class torch.tensor, size [3,64,64]
    print(photo[1])

    inter1 = analyze.linear_interpolate(photo[0], photo[1], model, device)
    print("interp1", type(inter1), [t.size() for t in inter1])
    for t in inter1:
        print(t.numpy().shape, type(t.numpy())) #numpy array of size (3, 64, 64) each
    #print(inter1)
    save_image(inter1, 'checkpoints/interpolate-dfc.png', padding=0, nrow=10)

if interpolating_song:

    folder_name = 'final_interpolation_dataset_unseen'
    song_name_1 = 'fe991a0954c5194c75037fe571061c6b_12490.midi_5_1.midi'
    song_name_2 = 'fe991a0954c5194c75037fe571061c6b_12490.midi_5_3.midi'
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
    state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
    state_1_last = state_1[0:42]
    state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
    state_2_last = state_2[0:42]
    #print(state_2[40])

    '''prove that is possible reshape [64, 64, 2] to [2, 64, 64] for feeding the NN
    and then reshaping to [64, 64, 2] in order to convert the tensor back to midi'''
    state_1 = np.einsum('ijk->kij', state_1)  # shape 2x64x64
    state_2 = np.einsum('ijk->kij', state_2)  # shape 2x64x64
    state_1 = state_1.astype(np.float32)  # set to float in order to keep data consistency
    state_2 = state_2.astype(np.float32)  # set to float in order to keep data consistency
    #print("reshaped to:", type(state_1), state_1.shape)
    #print(state_1)

    ''' input image to be encoded and decoded is
                torch tensor of [3, 64, 64] --> [78, 78, 2]'''

    '''
    print("original song shape:", type(state), np.asarray(state).shape)
    state = np.einsum('ijk->kij', np.asarray(state))
    print("reshaped song:", type(state), state.shape)
    state = torch.from_numpy(np.asarray(state)) # convert state: list to numpy array to torch tensor
    '''
    state_1 = torch.from_numpy(state_1)  # convert state: numpy array to torch tensor
    state_2 = torch.from_numpy(state_2)  # convert state: numpy array to torch tensor
    #print("numpy to torch tensor:")
    #print(state_1)
    #print("song to encode:", type(state_1), state_1.shape)

    inter1 = analyze.linear_interpolate(state_1, state_2, model, device)
    k = 0
    #total_np = np.empty()
    for t in inter1:
        print(k)
        print(t.numpy().shape, type(t.numpy())) #numpy array of size (3, 64, 64) each
        nump_song = t.numpy()
        #print("reshaped to:", type(nump_song), nump_song.shape)
        nump_song = np.einsum('kij->ijk', nump_song)  # shape 64x64x2 (as original)
        print("reshaped to:", type(nump_song), nump_song.shape)
        nump_song = nump_song[0:42]
        nump_song[nump_song >= .35] = 1
        nump_song[nump_song < .35] = 0
        #print(nump_song[40])
        if k==0:
            total_np = nump_song
        else:
            total_np = np.concatenate((total_np, nump_song), axis=0)
        print(total_np.shape)

        k+=1
    print("last:",total_np.shape)
    total_np = np.concatenate((state_1_last, total_np), axis=0)
    total_np = np.concatenate((total_np, state_2_last), axis=0)
    #print(total_np[30])
    noteStateMatrixToMidi(total_np, folder_name + '/' + song_name_1 + '_interp_' + str(k))  # save as midi

if interpolating_flow:
    folder_name = 'final_interpolation_dataset_unseen'
    song_name_1 = '4d304a5af6078632e2ea610a22c1f84d_2727.midi_5_1.midi'
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
    state_1 = padStateMatrix(folder_name, song_name_1, target_length, pad, save_as_midi=False)  # shape 64x64x2
    state_1_last = state_1[0:42]
    state_2 = padStateMatrix(folder_name, song_name_2, target_length, pad, save_as_midi=False)  # shape 64x64x2
    state_2_last = state_2[0:42]
    #print(state_2[40])
    state_1 = np.einsum('ijk->kij', state_1)  # shape 2x64x64
    state_2 = np.einsum('ijk->kij', state_2)  # shape 2x64x64
    state_1 = state_1.astype(np.float32)  # set to float in order to keep data consistency
    state_2 = state_2.astype(np.float32)  # set to float in order to keep data consistency
    state_1 = torch.from_numpy(state_1)  # convert state: numpy array to torch tensor
    state_2 = torch.from_numpy(state_2)  # convert state: numpy array to torch tensor

    model.eval()
    z1 = analyze.get_z(state_1, model, device)
    print("z1:", z1, z1.shape, type(z1))
    z1 = z1.numpy()
    print("z1:", z1, z1.shape, type(z1))
    z2 = analyze.get_z(state_2, model, device)
    z2 = z2.numpy()
    print("z2:", z2, z2.shape)

    matrix_1 = 0.5*np.ones((100,100))

    #factors = np.linspace(0.9, 0.1, num=1)  # 10, #numpy array [1, num] with ranges between 0~1
    factors = [0.5]
    # print(type(factors))
    print("factors", factors)
    result = []

    with torch.no_grad():

        for f in factors:
            #z = (f * z1 + (1 - f) * z2).to(device) # z class = torch tensor
            z = (np.matmul(z1, matrix_1) + np.matmul(z2, matrix_1))
            print("z:", z, z.shape, type(z))
            z = z.astype(np.float32)  # set to float in order to keep data consistency
            print("z:", z, z.shape, type(z))
            z = torch.from_numpy(z)
            print("z:", z, z.shape, type(z))
            im = torch.squeeze(model.decode(z))
            #im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    k = 0
    for t in result:
        print("k:", k)
        print(t.numpy().shape, type(t.numpy())) #numpy array of size (3, 64, 64) each
        nump_song = t.numpy()
        #print("reshaped to:", type(nump_song), nump_song.shape)
        nump_song = np.einsum('kij->ijk', nump_song)  # shape 64x64x2 (as original)
        print("reshaped to:", type(nump_song), nump_song.shape)
        nump_song = nump_song[0:42]
        nump_song[nump_song >= .35] = 1
        nump_song[nump_song < .35] = 0
        #print(nump_song[40])
        if k==0:
            total_np = nump_song
        else:
            total_np = np.concatenate((total_np, nump_song), axis=0)
        print(total_np.shape)

        k+=1

    print("last:",total_np.shape)
    total_np = np.concatenate((state_1_last, total_np), axis=0)
    total_np = np.concatenate((total_np, state_2_last), axis=0)
    print(total_np.shape)
    noteStateMatrixToMidi(total_np, folder_name + '/A_FLOW_' + song_name_1 + '_interp_' + str(k))  # save as midi



if plot_loss_plot:
    train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))
    #print(train_losses)
    #print(test_losses)
    train_x, train_l = zip(*train_losses)
    test_x, test_l = zip(*test_losses)
    print(train_x)
    print(len(train_x))
    print(train_l)
    print(len(train_l))
    print(len(test_x))
    print(len(test_l))
    '''
    for i in range(0,688):
        train_x = list(train_x)
        train_l = list(train_l)
        test_x = list(test_x)
        test_l = list(test_l)
        train_x.pop(0)
        train_l.pop(0)
        test_x.pop(0)
        test_l.pop(0)
    '''
    train_x = train_x[12143:16143] #+ train_x[10688:]
    train_l = train_l[12143:16143]  # + train_x[10688:]
    test_x = test_x[12143:16143]  # + train_x[10688:]
    test_l = test_l[12143:16143]  # + train_x[10688:]

    print(len(train_x))
    print(train_x)
    print(len(train_l))
    print(len(test_x))
    print(len(test_l))

    plt.figure()
    plt.title('Train Loss vs. Test Loss')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.plot(train_x, train_l, 'b', label='train_loss')
    plt.plot(test_x, test_l, 'r', label='test_loss')
    plt.legend()
    plt.show()
    #analyze.plot_loss(train_losses, test_losses, PLOT_PATH)