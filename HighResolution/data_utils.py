from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

import numpy as np

import tensorflow as tf












########################################################################
# FLAGS (options) for this app
########################################################################

tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")

tf.app.flags.DEFINE_integer("direction", 2, "Set to 0/1/2 for Forward/Backward/Bi-directional.")

tf.app.flags.DEFINE_boolean("use_intensity", True, "Set to True to use intensity-model.")

tf.app.flags.DEFINE_boolean("shared", False, "Set to True to use shared weights.")

tf.app.flags.DEFINE_boolean("use_lstm", True, "Set to True to use lstm-model.")

tf.app.flags.DEFINE_boolean("knapsack_build", False, "Set to True to build knapsack matrix.")

tf.app.flags.DEFINE_boolean("train", False, "Set to True for training.")

tf.app.flags.DEFINE_boolean("test_true_feeding", False, "Set to True for testing.")

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for decoding.")

tf.app.flags.DEFINE_boolean("beam_search", False, "Set to True for beam search.")

tf.app.flags.DEFINE_integer("beam_size", 1, "Number of optimal paths to search during decoding.")

FLAGS = tf.app.flags.FLAGS
########################################################################











########################################################################
# VOCABULARY 
########################################################################

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = ['A',
         'R',
         'N',
         'Nmod',
         'D',
         #~ 'C',
         'Cmod',
         'E',
         'Q',
         'Qmod',
         'G',
         'H',
         'I',
         'L',
         'K',
         'M',
         'Mmod',
         'F',
         'P',
         'S',
         'T',
         'W',
         'Y',
         'V',
        ]
#
vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)
#
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)
#
vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)





########################################################################
# MASS
########################################################################

mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD':0.0,
         '_GO':mass_N_terminus-mass_H,
         '_EOS':mass_C_terminus+mass_H,
         'A':71.03711, # 0
         'R':156.10111, # 1
         'N':114.04293, # 2
         'Nmod':115.02695,
         'D':115.02694, # 3
         #~ 'C':103.00919, # 4
         'Cmod':160.03065, # C(+57.02)
         #~ 'Cmod':161.01919, # C(+58.01) # orbi
         'E':129.04259, # 5
         'Q':128.05858, # 6
         'Qmod':129.0426,
         'G':57.02146, # 7
         'H':137.05891, # 8
         'I':113.08406, # 9
         'L':113.08406, # 10
         'K':128.09496, # 11
         'M':131.04049, # 12
         'Mmod':147.0354,
         'F':147.06841, # 13
         'P':97.05276, # 14
         'S':87.03203, # 15
         'T':101.04768, # 16
         'W':186.07931, # 17
         'Y':163.06333, # 18
         'V':99.06841, # 19
        }

mass_ID = [mass_AA[vocab_reverse[x]] for x in xrange(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146






########################################################################
# PRECISION & RESOLUTION & temp-Limits of MASS & LEN
########################################################################

# if change, need to re-compile cython_speedup
#~ SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
#~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
#~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
SPECTRUM_RESOLUTION = 50 # bins for 1.0 Da = precision 0.02 Da
#~ SPECTRUM_RESOLUTION = 80 # bins for 1.0 Da = precision 0.0125 Da
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# if change, need to re-compile cython_speedup
WINDOW_SIZE = 10 # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

MZ_MAX = 3000.0
MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION) # 30k

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.0215 # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
AA_MATCH_PRECISION = 0.1

MAX_LEN = 30
if (FLAGS.decode): # for decode 
  MAX_LEN = 50
print("MAX_LEN ", MAX_LEN)

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [12,22,32] 
print("_buckets ", _buckets)






########################################################################
# TRAINING PARAMETERS
########################################################################

num_ion = 8 # 2
print("num_ion ", num_ion)

l2_loss_weight = 0.0 # 0.0
print("l2_loss_weight ", l2_loss_weight)

#~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
#~ encoding_cnn_filter = 4
#~ print("encoding_cnn_size ", encoding_cnn_size)
#~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_layers = 1
num_units = 512
print("num_layers ", num_layers)
print("num_units ", num_units)

keep_conv = 0.75
keep_dense = 0.5
print("keep_conv ", keep_conv)
print("keep_dense ", keep_dense)

batch_size = 128
print("batch_size ", batch_size)

epoch_stop = 20
print("epoch_stop ", epoch_stop)

train_stack_size = 4500
valid_stack_size = 9000
test_stack_size = 4000
print("train_stack_size ", train_stack_size)
print("valid_stack_size ", valid_stack_size)
print("test_stack_size ", test_stack_size)

steps_per_checkpoint = 100
random_test_batches = 10
print("steps_per_checkpoint ", steps_per_checkpoint)
print("random_test_batches ", random_test_batches)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)






########################################################################
# DATASETS
########################################################################

# CROSS-9HIGH_80k.EXCLUDE_BACILLUS-REPEAT
data_format = "mgf"
input_file_train = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.train.repeat"
input_file_valid = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.valid.repeat"
input_file_test = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.test.repeat"
input_file_test = "data/high.bacillus.PXD004565/peaks.db.10k.mgf"
decode_test_file = "data/high.bacillus.PXD004565/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_CLAMBACTERIA-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.clambacteria.PXD004536/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.clambacteria.PXD004536/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_HONEYBEE-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.honeybee.PXD004467/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.honeybee.PXD004467/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_HUMAN-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_human/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_human/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_human/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.human.PXD004424/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.human.PXD004424/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_MOUSE-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_mouse/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_mouse/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_mouse/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.mouse.PXD004948/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.mouse.PXD004948/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_MMAZEI-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.mmazei.PXD004325/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.mmazei.PXD004325/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_RICEBEAN-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.ricebean.PXD005025/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.ricebean.PXD005025/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_TOMATO-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_tomato/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_tomato/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_tomato/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.tomato.PXD004947/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.tomato.PXD004947/peaks.db.10k.mgf"

# CROSS-9HIGH_80k.EXCLUDE_YEAST-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.9high_80k.exclude_yeast/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.9high_80k.exclude_yeast/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.9high_80k.exclude_yeast/cross.cat.mgf.test.repeat"
#~ input_file_test = "data/high.yeast.PXD003868/peaks.db.10k.mgf"
#~ decode_test_file = "data/high.yeast.PXD003868/peaks.db.10k.mgf"






# DATA-test

#~ decode_test_file = "data/high.bacillus.PXD004565/peaks.db.mgf"
#~ decode_test_file = "data/high.clambacteria.PXD004536/peaks.db.mgf"
#~ decode_test_file = "data/high.honeybee.PXD004467/peaks.db.mgf"
#~ decode_test_file = "data/high.human.PXD004424/peaks.db.mgf"
#~ decode_test_file = "data/high.mmazei.PXD004325/peaks.db.mgf"
#~ decode_test_file = "data/high.mouse.PXD004948/peaks.db.mgf"
#~ decode_test_file = "data/high.ricebean.PXD005025/peaks.db.mgf"
#~ decode_test_file = "data/high.tomato.PXD004947/peaks.db.mgf"
#~ decode_test_file = "data/high.yeast.PXD003868/peaks.db.mgf"
########################################################################
























