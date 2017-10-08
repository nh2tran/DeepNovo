# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
#
# The source code in this file originated from the sequence-to-sequence tutorial
# of TensorFlow, Google Inc. I have modified the entire code to solve the 
# problem of peptide sequencing. The copyright notice of Google is attached 
# above as required by its Apache License.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import deepnovo_config
import deepnovo_model_training
import deepnovo_model_decoding


class TrainingModel(object):
  """TODO(nh2tran): docstring."""

  def __init__(self, session, training_mode): # TODO(nh2tran): session-unused
    """TODO(nh2tran): docstring."""

    print("TrainingModel: __init__()")

    self.global_step = tf.Variable(0, trainable=False)

    # Dropout probabilities
    self.keep_conv_holder = tf.placeholder(dtype=tf.float32, name="keep_conv")
    self.keep_dense_holder = tf.placeholder(dtype=tf.float32, name="keep_dense")

    # INPUT PLACEHOLDERS

    # spectrum
    self.encoder_inputs = [tf.placeholder(dtype=tf.float32,
                                          shape=[None, deepnovo_config.MZ_SIZE],
                                          name="encoder_inputs")]

    # candidate intensity
    self.intensity_inputs_forward = []
    self.intensity_inputs_backward = []
    for x in xrange(deepnovo_config._buckets[-1]): # TODO(nh2tran): _buckets
      self.intensity_inputs_forward.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None, deepnovo_config.vocab_size, deepnovo_config.num_ion, deepnovo_config.WINDOW_SIZE], # TODO(nh2tran): line-too-long, config
          name="intensity_inputs_forward_{0}".format(x)))
      self.intensity_inputs_backward.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None, deepnovo_config.vocab_size, deepnovo_config.num_ion, deepnovo_config.WINDOW_SIZE], # TODO(nh2tran): line-too-long, config
          name="intensity_inputs_backward_{0}".format(x)))

    # decoder inputs
    self.decoder_inputs_forward = []
    self.decoder_inputs_backward = []
    self.target_weights = []
    for x in xrange(deepnovo_config._buckets[-1] + 1): # TODO(nh2tran): _buckets
      self.decoder_inputs_forward.append(tf.placeholder(
          dtype=tf.int32,
          shape=[None],
          name="decoder_inputs_forward_{0}".format(x)))
      self.decoder_inputs_backward.append(tf.placeholder(
          dtype=tf.int32,
          shape=[None],
          name="decoder_inputs_backward_{0}".format(x)))
      self.target_weights.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None],
          name="target_weights_{0}".format(x)))

    # Our targets are decoder inputs shifted by one.
    self.targets_forward = [self.decoder_inputs_forward[x + 1]
                            for x in xrange(len(self.decoder_inputs_forward) - 1)] # TODO(nh2tran): line-too-long
    self.targets_backward = [self.decoder_inputs_backward[x + 1]
                             for x in xrange(len(self.decoder_inputs_backward) - 1)] # TODO(nh2tran): line-too-long

    # OUTPUTS and LOSSES
    (self.outputs_forward,
     self.outputs_backward,
     self.losses) = deepnovo_model_training.train(self.encoder_inputs,
                                           self.intensity_inputs_forward,
                                           self.intensity_inputs_backward,
                                           self.decoder_inputs_forward,
                                           self.decoder_inputs_backward,
                                           self.targets_forward,
                                           self.targets_backward,
                                           self.target_weights,
                                           self.keep_conv_holder,
                                           self.keep_dense_holder)

    # Gradients and SGD update operation for training the model.
    if training_mode:
      params = tf.trainable_variables()
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.AdamOptimizer()
      for b in xrange(len(deepnovo_config._buckets)): # TODO(nh2tran): _buckets
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            gradients,
            deepnovo_config.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params),
            global_step=self.global_step))

      # for TensorBoard
      #~ self.train_writer = tf.train.SummaryWriter(deepnovo_config.FLAGS.train_dir, session.graph)
      #~ self.loss_summaries = [tf.scalar_summary("losses_" + str(b), self.losses[b])
                             #~ for b in xrange(len(deepnovo_config._buckets))]
      #~ dense1_W_penalty = tf.get_default_graph().get_tensor_by_name(
                         #~ "model_with_buckets/embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/dense1_W_penalty:0")
      #~ self.dense1_W_penalty_summary = tf.scalar_summary("dense1_W_penalty_summary", dense1_W_penalty)

    # Saver
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

  def step(self,
           session,
           encoder_inputs,
           intensity_inputs_forward=None,
           intensity_inputs_backward=None,
           decoder_inputs_forward=None,
           decoder_inputs_backward=None,
           target_weights=None,
           bucket_id=0,
           training_mode=True):
    """TODO(nh2tran): docstring."""

    # Check if the sizes match.
    decoder_size = deepnovo_config._buckets[bucket_id] # TODO(nh2tran): _buckets

    # Input feed
    input_feed = {}
    input_feed[self.encoder_inputs[0].name] = encoder_inputs[0]

    # Input feed forward
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_forward[x].name] = intensity_inputs_forward[x] # TODO(nh2tran): line-too-long
        input_feed[self.decoder_inputs_forward[x].name] = decoder_inputs_forward[x] # TODO(nh2tran): line-too-long
      # Since our targets are decoder inputs shifted by one, we need one more.
      last_target_forward = self.decoder_inputs_forward[decoder_size].name
      input_feed[last_target_forward] = np.zeros([encoder_inputs[0].shape[0]],
                                                 dtype=np.int32)

    # Input feed backward
    if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_backward[x].name] = intensity_inputs_backward[x] # TODO(nh2tran): line-too-long
        input_feed[self.decoder_inputs_backward[x].name] = decoder_inputs_backward[x] # TODO(nh2tran): line-too-long
      # Since our targets are decoder inputs shifted by one, we need one more.
      last_target_backward = self.decoder_inputs_backward[decoder_size].name
      input_feed[last_target_backward] = np.zeros([encoder_inputs[0].shape[0]],
                                                  dtype=np.int32)

    # Input feed target weights
    for x in xrange(decoder_size):
      input_feed[self.target_weights[x].name] = target_weights[x]

    # keeping probability for dropout layers
    if training_mode:
      input_feed[self.keep_conv_holder.name] = deepnovo_config.keep_conv
      input_feed[self.keep_dense_holder.name] = deepnovo_config.keep_dense
    else:
      input_feed[self.keep_conv_holder.name] = 1.0
      input_feed[self.keep_dense_holder.name] = 1.0

    # Output feed: depends on whether we do a back-propagation
    if training_mode:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.

    # Output forward logits
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        output_feed.append(self.outputs_forward[bucket_id][x])

    # Output backward logits
    if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        output_feed.append(self.outputs_backward[bucket_id][x])

    # RUN
    outputs = session.run(fetches=output_feed, feed_dict=input_feed)

    # for TensorBoard
    #~ if (training_mode and (self.global_step.eval() % deepnovo_config.steps_per_checkpoint == 0)):
      #~ summary_op = tf.merge_summary([self.loss_summaries[bucket_id], self.dense1_W_penalty_summary])
      #~ summary_str = session.run(summary_op, feed_dict=input_feed)
      #~ self.train_writer.add_summary(summary_str, self.global_step.eval())

    if training_mode:
      # Gradient norm, loss, [outputs_forward, outputs_backward]
      if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
        return outputs[1], outputs[2], outputs[3:]
      else:
        return outputs[1], outputs[2], outputs[3:(3+decoder_size)], outputs[(3+decoder_size):] # TODO(nh2tran): line-too-long
    else:
      # No gradient norm, loss, [outputs_forward, outputs_backward]
      if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
        return None, outputs[0], outputs[1:]
      else:
        return None, outputs[0], outputs[1:(1+decoder_size)], outputs[(1+decoder_size):] # TODO(nh2tran): line-too-long


class DecodingModel(object):
  """TODO(nh2tran): docstring."""

  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("DecodingModel: __init__()")

    # INPUT placeholder
    self.input_spectrum = tf.placeholder(dtype=tf.float32,
                                         shape=[None, deepnovo_config.MZ_SIZE],
                                         name="input_spectrum")
    # nobi
    self.input_AA_id = [tf.placeholder(dtype=tf.int32,
                                       shape=[None],
                                       name="input_AA_id_1"),
                        tf.placeholder(dtype=tf.int32,
                                       shape=[None],
                                       name="input_AA_id_2")]
    self.input_intensity = tf.placeholder(dtype=tf.float32,
                                          shape=[None,
                                                 deepnovo_config.vocab_size,
                                                 deepnovo_config.num_ion,
                                                 deepnovo_config.WINDOW_SIZE],
                                          name="input_intensity")

    #~ self.input_state = (tf.placeholder(tf.float32,shape=[None,cell.state_size],name="input_state"),
    self.input_state = (tf.placeholder(dtype=tf.float32,
                                       shape=[None, deepnovo_config.num_units],
                                       name="input_c_state"),
                        tf.placeholder(dtype=tf.float32,
                                       shape=[None, deepnovo_config.num_units],
                                       name="input_h_state"))

    # OUTPUT
    ((self.lstm_state0_forward,
      self.output_log_prob_forward,
      self.lstm_state_forward),
     (self.lstm_state0_backward,
      self.output_log_prob_backward,
      self.lstm_state_backward)) = deepnovo_model_decoding.decode(self.input_spectrum,
                                                           self.input_AA_id,
                                                           self.input_intensity,
                                                           #~ self.input_state[0],
                                                           self.input_state)


  def restore(self, session):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("DecodingModel: restore()")

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(deepnovo_config.FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
      print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Error: model not found.")
      sys.exit()

