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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope

import deepnovo_config
import deepnovo_model_training


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


class ModelNetwork(object):
  """TODO(nh2tran): docstring.
     Core neural networks to calculate the probability of the next amino acid.
  """

  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.MZ_SIZE = deepnovo_config.MZ_SIZE
    self.vocab_size = deepnovo_config.vocab_size
    self.num_ion = deepnovo_config.num_ion
    self.WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
    self.num_units = deepnovo_config.num_units
    self.embedding_size = deepnovo_config.embedding_size
    self.num_layers = deepnovo_config.num_layers
    self.use_ion = deepnovo_config.FLAGS.use_intensity # TODO(nh2tran): change to "use_ion"
    self.use_lstm = deepnovo_config.FLAGS.use_lstm

    # keep_prob probability of dropout layers, will be defined in build()
    self.dropout_keep = None


  def build_network(self, input_dict, dropout_keep):
    """TODO(nh2tran): docstring.
       Build neural networks to calculate the probability of the next amino acid.

       Inputs:
         Input tensors are grouped into a dictionary.
         input_dict["spectrum"]: 2D tensor of shape [batch_size, MZ_SIZE].
         input_dict["intensity"]: [batch_size, vocab_size, num_ion, WINDOW_SIZE].
         input_dict["lstm_state"]: tuple of 2 tensors [batch_size, num_units]
         input_dict["AAid"]: list of 2 tensors [batch_size]
         dropout_keep["conv"]: keep_prob of dropout after convolutional layers
         dropout_keep["dense"]: keep_prob of dropout after dense layers

       Outputs:
         Output tensors are grouped into 2 dictionaries, output_forward and
         output_backward, each has 4 tensors:
         ["logit"]: [batch_size, vocab_size], to compute loss in training
         ["logprob"]: [batch_size, vocab_size], to compute score in inference
         ["lstm_state"]: [batch_size, num_units], to compute next iteration
         ["lstm_state0"]: [batch_size, num_units], state from cnn_spectrum
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: build_network()")

    self.dropout_keep = dropout_keep

    cnn_spectrum_feature = self._build_cnn_spectrum(input_dict["spectrum"])
    embedding_AAid = self._build_embedding_AAid(input_dict["AAid"])

    output_forward = {}
    output_backward = {}
    for direction, output in zip(["forward", "backward"],
                                 [output_forward, output_backward]):

      scope = "embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder" # TODO(nh2tran): change to "cnn_ion/lstm"
      scope = scope + "_" + direction
      with tf.variable_scope(scope):

        # cnn_ion model
        cnn_ion_feature, cnn_ion_logit = self._build_cnn_ion(
            input_dict["intensity"],
            direction)

        # lstm model
        lstm_feature, lstm_logit, lstm_state0, lstm_state = self._build_lstm(
            cnn_spectrum_feature,
            input_dict["lstm_state"],
            embedding_AAid,
            direction)

        # combine cnn_ion and lstm features
        feature_weight = tf.get_variable(
            name="dense_concat_W", # TODO(nh2tran): change to "feature_weight"
            shape=[self.num_units*2, self.num_units],
            initializer=tf.uniform_unit_scaling_initializer(1.43))
        feature_bias = tf.get_variable(
            name="dense_concat_B", # TODO(nh2tran): change to "feature_bias"
            shape=[self.num_units],
            initializer=tf.constant_initializer(0.1))
        feature_input = tf.concat(values=[cnn_ion_feature, lstm_feature],
                                  axis=1)
        feature = tf.nn.relu(tf.matmul(feature_input, feature_weight)
                             + feature_bias)
        feature = tf.nn.dropout(feature, self.dropout_keep["dense"])
  
        # linear transform to logit [128, 26]
        # TODO(nh2tran): replace _linear and remove scope
        with tf.variable_scope("output_logit"):
          feature_logit = rnn_cell_impl._linear(args=feature,
                                                output_size=self.vocab_size,
                                                bias=True,
                                                bias_initializer=None,#0.1,
                                                kernel_initializer=None)

        # both ion-lstm models are used together by default
        # but each can be used separately for investigation
        if self.use_ion and self.use_lstm:
          logit = feature_logit
        elif self.use_ion:
          logit = cnn_ion_logit
        elif self.use_lstm:
          logit = lstm_logit
        else:
          print("Error: wrong ion-lstm model!")
          sys.exit()

        logprob = tf.log(tf.nn.softmax(logit))

        output["logit"] = logit
        output["logprob"] = logprob
        output["lstm_state"] = lstm_state
        output["lstm_state0"] = lstm_state0

    return output_forward, output_backward


  def _build_cnn_ion(self, input_intensity, direction):
    """TODO(nh2tran): docstring.

       Inputs:
         input_intensity: shape [batch_size, vocab_size, num_ion, WINDOW_SIZE].
         direction: "forward" or "backward".

       Outputs:
         cnn_ion: shape [batch_size, num_units]
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: _build_cnn_ion()")

    # reshape [128, 26, 8, 10] to [128, 8, 10, 26] to do convolution along the
    #   window_size dimension.
    # TODO(nh2tran): this can be fixed at the input process.
    input_intensity = tf.transpose(input_intensity, perm=[0, 2, 3, 1])

    # conv1: [128, 8, 10, 26] >> [128, 8, 10, 64] with kernel [1, 3, 26, 64]
    conv1_weight = tf.get_variable(
        name="conv1_weights", # TODO(nh2tran): to change to "conv1_weight"
        shape=[1, 3, self.vocab_size, 64],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv1_bias = tf.get_variable(
        name="conv1_biases", # TODO(nh2tran): to change to "conv1_bias"
        shape=[64],
        initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.relu(tf.nn.conv2d(input_intensity,
                                    conv1_weight,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                       + conv1_bias)

    # conv2: [128, 8, 10, 64] >> [128, 8, 10, 64] with kernel [1, 2, 64, 64]
    conv2_weight = tf.get_variable(
        name="conv2_weights", # TODO(nh2tran): change to "conv2_weight"
        shape=[1, 2, 64, 64],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv2_bias = tf.get_variable(
        name="conv2_biases", # TODO(nh2tran): change to "conv2_bias"
        shape=[64],
        initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                    conv2_weight,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                       + conv2_bias)
    # max pooling [1, 1, 3, 1] with stride [1, 1, 2, 1]
    conv2 = tf.nn.max_pool(conv2,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 2, 1],
                           padding='SAME') # [128,8,10,64]
    conv2 = tf.nn.dropout(conv2, self.dropout_keep["conv"])

    # dense1: 4D >> [128, 512]
    dense1_input_size = self.num_ion * (self.WINDOW_SIZE // 2) * 64
    dense1_output_size = self.num_units
    dense1_weight = tf.get_variable(
        name="dense1_weights", # TODO(nh2tran): change to "dense1_weight"
        shape=[dense1_input_size, dense1_output_size],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense1_bias = tf.get_variable(
        name="dense1_biases", # TODO(nh2tran): change to "dense1_bias"
        shape=[dense1_output_size],
        initializer=tf.constant_initializer(0.1))
    dense1_input = tf.reshape(conv2, [-1, dense1_input_size])
    dense1 = tf.nn.relu(tf.matmul(dense1_input, dense1_weight) + dense1_bias)
    dense1 = tf.nn.dropout(dense1, self.dropout_keep["dense"], name="dropout1") # TODO(nh2tran): remove name

    cnn_ion_feature = dense1

    # linear transform to logit [128, 26], in case only cnn_ion model is used
    # TODO(nh2tran): replace _linear and remove scope
    with tf.variable_scope("intensity_output_projected"):
      cnn_ion_logit = rnn_cell_impl._linear(args=cnn_ion_feature,
                                            output_size=self.vocab_size,
                                            bias=True,
                                            bias_initializer=None,#0.1,
                                            kernel_initializer=None)
    

    return cnn_ion_feature, cnn_ion_logit


  def _build_cnn_spectrum(self, input_spectrum):
    """TODO(nh2tran): docstring.

       Inputs:
         input_spectrum: 2D tensor of shape [batch_size, MZ_SIZE].

       Outputs:
         cnn_spectrum_feature: 2D tensor of shape [batch_size, num_units]
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: _build_cnn_spectrum()")

    scope = "embedding_rnn_seq2seq" # TODO(nh2tran): change to "cnn_spectrum"
    with tf.variable_scope(scope):
  
      # reshape the 2D input tensor to common 4D
      layer0 = tf.reshape(input_spectrum, [-1, 1, self.MZ_SIZE, 1])
  
      # conv1: filter [1, 4, 1, 4] with stride [1, 1, 1, 1]
      conv1_weight = tf.get_variable(
          name="conv1_W", # TODO(nh2tran): change to "conv1_weight"
          shape=[1, 4, 1, 4],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv1_bias = tf.get_variable(
          name="conv1_B", # TODO(nh2tran): change to "conv1_bias"
          shape=[4],
          initializer=tf.constant_initializer(0.1))
      conv1 = tf.nn.relu(tf.nn.conv2d(layer0,
                                      conv1_weight,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + conv1_bias)

      # conv2: filter [1, 4, 4, 4] with stride [1, 1, 1, 1]
      conv2_weight = tf.get_variable(
          name="conv2_W", # TODO(nh2tran): change to "conv2_weight"
          shape=[1, 4, 4, 4],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv2_bias = tf.get_variable(
          name="conv2_B", # TODO(nh2tran): change to "conv2_bias"
          shape=[4],
          initializer=tf.constant_initializer(0.1))
      conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                      conv2_weight,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + conv2_bias)
      # max pooling [1, 1, 6, 1] with stride [1, 1, 4, 1]
      conv2 = tf.nn.max_pool(conv2,
                             ksize=[1, 1, 6, 1],
                             strides=[1, 1, 4, 1],
                             padding='SAME')
      conv2 = tf.nn.dropout(conv2, self.dropout_keep["conv"])

      # dense1
      dense1_input_size = 1 * (self.MZ_SIZE // (4)) * 4
      dense1_output_size = self.num_units
      dense1_weight = tf.get_variable(
          name="dense1_W", # TODO(nh2tran): change to "dense1_weight"
          shape=[dense1_input_size, dense1_output_size],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_bias = tf.get_variable(
          name="dense1_B", # TODO(nh2tran): change to "dense1_bias"
          shape=[dense1_output_size],
          initializer=tf.constant_initializer(0.1))
      dense1 = tf.reshape(conv2, [-1, dense1_input_size])
      dense1 = tf.nn.relu(tf.matmul(dense1, dense1_weight) + dense1_bias)
      dense1 = tf.nn.dropout(dense1, self.dropout_keep["dense"])

      cnn_spectrum_feature = dense1

    return cnn_spectrum_feature


  def _build_embedding_AAid(self, input_AAid):
    """TODO(nh2tran): docstring.

       Inputs:
         input_AAid: list of 2 tensors [batch_size].

       Outputs:
         embedding_AAid: list of 2 tensors [batch_size, embedding_size].
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: _build_embedding_AAid()")

    scope = "embedding_rnn_seq2seq/embedding_rnn_decoder" # TODO(nh2tran): to change to "embedding_AAid"
    with tf.variable_scope(scope):

      with ops.device("/cpu:0"):
        embedding = tf.get_variable(
            name="embedding",
            shape=[self.vocab_size, self.embedding_size])

      embedding_AAid = [embedding_ops.embedding_lookup(embedding, x)
                        for x in input_AAid]

    return embedding_AAid


  def _build_lstm(self, cnn_spectrum, input_lstm_state, embedding_AAid, direction):
    """TODO(nh2tran): docstring.

       Inputs:
         cnn_spectrum: shape [batch_size, num_units].
         input_lstm_state: tuple of 2 tensors [batch_size, num_units].
         embedding_AAid: list of 2 tensors [batch_size, embedding_size].
         direction: "forward" or "backward".

       Outputs:
         lstm_output: shape [batch_size, num_units].
         lstm_state0: tuple of 2 tensors [batch_size, num_units].
         lstm_state2: tuple of 2 tensors [batch_size, num_units].
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: _build_lstm()")

    # BUG rnn_cell tf.1.x: use separate BasicLSTMCell for 2 directions. Ok, fixed.
    single_cell = rnn_cell.BasicLSTMCell(num_units=self.num_units,
                                         state_is_tuple=True)
    if self.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
    else:
      cell = single_cell
    cell = rnn_cell.DropoutWrapper(cell,
                                   input_keep_prob=self.dropout_keep["dense"],
                                   output_keep_prob=self.dropout_keep["dense"])

    # project lstm input from embedding_size to num_units
    with tf.variable_scope("LSTM_input_projected"): # TODO(nh2tran): remove

      project_weight = tf.get_variable(
          name="lstm_input_projected_W", # TODO(nh2tran): change to "project_weight"
          shape=[self.embedding_size, self.num_units])
      project_bias = tf.get_variable(
          name="lstm_input_projected_B", # TODO(nh2tran): change to "project_bias"
          shape=[self.num_units],
          initializer=tf.constant_initializer(0.1))
      # nobi
      AA_1 = embedding_AAid[0]
      AA_2 = embedding_AAid[1]
      AA_1_project = (tf.matmul(AA_1, project_weight) + project_bias)
      AA_2_project = (tf.matmul(AA_2, project_weight) + project_bias)

    # lstm cell's one-iteration
    with tf.variable_scope("LSTM_cell"): # TODO(nh2tran): remove

      # cnn_spectrum as input 0 to initialize the lstm cell
      # lstm_state0 is returned for 2 purposes:
      #   (i) initializing several spectra in batch is faster
      #   (ii) using lstm on short 3-mers (nobi model)
      input0 = cnn_spectrum
      batch_size = array_ops.shape(input0)[0]
      zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      # nobi
      _, lstm_state0 = cell(inputs=input0, state=zero_state)

      # nobi model: use lstm on short 3-mers
      #~ _, lstm_state1 = cell(inputs=AA_1_project, state=input_lstm_state)
      #~ lstm_output, lstm_state2 = cell(inputs=AA_2_project, state=lstm_state1)
      # lstm.len_full model: standard lstm
      lstm_feature, lstm_state = cell(inputs=AA_2_project, state=input_lstm_state)

    # linear transform to logit [128, 26], in case only lstm model is used
    # TODO(nh2tran): replace _linear and remove scope
    with tf.variable_scope("lstm_output_projected"):
      lstm_logit = rnn_cell_impl._linear(
          args=lstm_feature,
          output_size=self.vocab_size,
          bias=True,
          bias_initializer=None,#0.1,
          kernel_initializer=None)

    return lstm_feature, lstm_logit, lstm_state0, lstm_state


class ModelInference(object):
  """TODO(nh2tran): docstring."""

  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.MZ_SIZE = deepnovo_config.MZ_SIZE
    self.vocab_size = deepnovo_config.vocab_size
    self.num_ion = deepnovo_config.num_ion
    self.WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
    self.num_units = deepnovo_config.num_units
    self.train_dir = deepnovo_config.FLAGS.train_dir

    # input tensors are grouped into a dictionary
    self.input_dict = {}
    # input spectrum is a 2D tensor of shape [batch_size, MZ_SIZE]
    #   for example: [128, 30k]
    self.input_dict["spectrum"] = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self.MZ_SIZE],
                                                 name="input_spectrum")
    # input intensity profile: [batch_size, vocab_size, num_ion, WINDOW_SIZE]
    #   for example; [128, 26, 8, 10]
    self.input_dict["intensity"] = tf.placeholder(
        dtype=tf.float32,
        shape=[None, self.vocab_size, self.num_ion, self.WINDOW_SIZE],
        name="input_intensity")
    # input lstm state is a tuple of 2 tensors [batch_size, num_units]
    #   for example: [128, 512]
    self.input_dict["lstm_state"] = (tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.num_units],
                                                   name="input_c_state"), # to change to "input_lstm_state_c"
                                    tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.num_units],
                                                   name="input_h_state")) # to change to "input_lstm_state_h"
    # input last 2 amino acids if using lstm for short 3-mers
    #   list of 2 tensors [batch_size]
    #   "AAid" stands for amino acid id
    self.input_dict["AAid"] = [tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name="input_AA_id_1"), # to change to "input_AAid_1"
                               tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name="input_AA_id_2")] # to change to "input_AAid_2"

    # the keep_prob probability of dropout layers
    #   for inference model, they are const 1.0
    #   for train/valid model, they are input tensors
    self.dropout_keep = {}
    self.dropout_keep["conv"] = 1.0
    self.dropout_keep["dense"] = 1.0

    # core neural networks to calculate output tensors from the input
    self.model_network = ModelNetwork()

    # output tensors are grouped into 2 dictionaries, forward and backward,
    #   each has 4 tensors:
    #   ["logit"]: shape [batch_size, vocab_size], to compute loss in training
    #   ["logprob"]: shape [batch_size, vocab_size], to compute score in inference
    #   ["lstm_state"]: shape [batch_size, num_units], to compute next iteration
    #   ["lstm_state0"]: shape [batch_size, num_units], state from cnn_spectrum
    # they will be built and loaded by build_model() and restore_model()
    self.output_forward = None
    self.output_backward = None


  def build_model(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: build_model()")

    self.output_forward, self.output_backward = self.model_network.build_network(
        self.input_dict,
        self.dropout_keep)


  def restore_model(self, session):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: restore_model()")

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(self.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
      print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Error: model not found.")
      sys.exit()

