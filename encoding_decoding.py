from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from magenta import music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `interpolate`).')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

'''
--config=cat-mel_2bar_big \
--checkpoint_file=tmp/train/model.ckpt-54029 \
--mode=interpolate \
--num_outputs=5 \
--input_midi_1=tmp/interpolation/begin1.mid \
--input_midi_2=tmp/interpolation/end1.mid \
--output_dir=tmp/interpolation/generated_big_lahk_reduced
'''
# --------------------------------------------------------------
# ------------------------- I N P U T --------------------------
# --------------------------------------------------------------
date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
config_map = configs.CONFIG_MAP
config = config_map[FLAGS.config]
config.data_converter.max_tensors_per_item = None

# LOADING FILES TO ENCODE
if FLAGS.input_midi_1 is None:
    raise ValueError(
        '`--input_midi_1` must be specified in '
        '`interpolate` mode.')
input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
#input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
if not os.path.exists(input_midi_1):
    raise ValueError('Input MIDI 1 not found: %s' % FLAGS.input_midi_1)
#if not os.path.exists(input_midi_2):
#    raise ValueError('Input MIDI 2 not found: %s' % FLAGS.input_midi_2)
input_1 = mm.midi_file_to_note_sequence(input_midi_1)
#input_2 = mm.midi_file_to_note_sequence(input_midi_2)

# LOADING MODEL
logging.info('Loading model...')
if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
    #checkpoint_dir_or_path = checkpoint_file
model = TrainedModel(
    config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
    checkpoint_dir_or_path=checkpoint_dir_or_path)

"""
Encodes a collection of NoteSequences into latent vectors.
    Args:
      note_sequences: A collection of NoteSequence objects to encode.
      assert_same_length: Whether to raise an AssertionError if all of the
        extracted sequences are not the same length.
    Returns:
      The encoded `z`, `mu`, and `sigma` values. (as tuple)
"""
logging.info('Encoding...')
#_, mu, _ = model.encode([input_1, input_2])
z, mu, sigma = model.encode([input_1])
#z = np.array([ # z = collection of latent vectors to decode
#    _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, FLAGS.num_outputs)]) #Spherical linear interpolation

results = model.decode(
    length=config.hparams.max_seq_len,
    z=z,
    temperature=FLAGS.temperature)

basename = os.path.join(
    FLAGS.output_dir,
    '%s_%s_%s-*-of-%03d.mid' %
    (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
for i, ns in enumerate(results):
    mm.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

logging.info('Done.')
