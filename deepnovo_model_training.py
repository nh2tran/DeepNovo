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

# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope

import deepnovo_config


def sequence_loss_per_sample(logits,
                             targets,
                             weights):
  """TODO(nh2tran): docstring.
  Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  #~ with tf.name_scope(name="sequence_loss_by_example",
                     #~ values=logits + targets + weights):
  with ops.op_scope(logits + targets + weights,
                    None,
                    "sequence_loss_by_example"):

    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      target = array_ops.reshape(math_ops.to_int64(target), [-1])
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit,
                                                                 labels=target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)

    # average_across_timesteps:
    total_size = math_ops.add_n(weights)
    total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
    log_perps /= total_size

  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  name):
  """TODO(nh2tran): docstring.
  Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  #~ with tf.name_scope(name=name,
                     #~ values=logits + targets + weights):
  with ops.op_scope(logits + targets + weights, name):
    cost = math_ops.reduce_sum(sequence_loss_per_sample(logits,
                                                        targets,
                                                        weights))
    batch_size = array_ops.shape(targets[0])[0]
    return cost / math_ops.cast(batch_size, dtypes.float32)


def decode_spectrum(encoded_spectrum,
                    intensity_inputs,
                    decoder_inputs_emb,
                    keep_conv,
                    keep_dense,
                    scope):
  """TODO(nh2tran): docstring.
  RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x cell.input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x cell.output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """

  single_cell = rnn_cell.BasicLSTMCell(num_units=deepnovo_config.num_units,
                                       state_is_tuple=True)
  #~ single_cell = rnn_cell.BasicRNNCell(num_units=deepnovo_config.num_units)
  #~ single_cell = rnn_cell.GRUCell(num_units=deepnovo_config.num_units)
  if deepnovo_config.num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * deepnovo_config.num_layers)
  else:
    cell = single_cell
  cell = rnn_cell.DropoutWrapper(cell,
                                 input_keep_prob=keep_dense,
                                 output_keep_prob=keep_dense)

  with variable_scope.variable_scope(scope):

    # INTENSITY-Model Parameters
    # intensity input [128, 27, 2, 10]

    if deepnovo_config.FLAGS.shared: # shared-weight

      dense1_input_size = deepnovo_config.num_ion * deepnovo_config.WINDOW_SIZE
      dense1_output_size = 1024
      dense1_W = variable_scope.get_variable(
          name="dense1_W_0",
          shape=[dense1_input_size, dense1_output_size],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_B = variable_scope.get_variable(
          name="dense1_B_0",
          shape=[dense1_output_size],
          initializer=tf.constant_initializer(0.1))

      dense_linear_W = variable_scope.get_variable(
          name="dense_linear_W",
          shape=[dense1_output_size, 1])
      dense_linear_B = variable_scope.get_variable(
          name="dense_linear_B",
          shape=[1],
          initializer=tf.constant_initializer(0.1))

    else: # joint-weight

      # conv1: [128, 8, 20, 26] >> [128, 8, 20, 64] with kernel [1, 3, 26, 64]
      conv1_weights = tf.get_variable(
          name="conv1_weights",
          shape=[1, 3, deepnovo_config.vocab_size, 64],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv1_biases = tf.get_variable(name="conv1_biases",
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.1))

      # conv2: [128, 8, 20, 64] >> [128, 8, 20, 64] with kernel [1, 2, 64, 64]
      conv2_weights = tf.get_variable(
          name="conv2_weights",
          shape=[1, 2, 64, 64],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv2_biases = tf.get_variable(name="conv2_biases",
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.1))

      # max_pool: [128, 8, 20, 64] >> [128, 8, 10, 64]

      # dense1: # 4D >> [128, 512]
      dense1_input_size = deepnovo_config.num_ion * (deepnovo_config.WINDOW_SIZE // 2) * 64 # deepnovo_config.vocab_size
      dense1_output_size = 512
      dense1_weights = tf.get_variable(
          "dense1_weights",
          shape=[dense1_input_size, dense1_output_size],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_biases = tf.get_variable("dense1_biases",
                                      shape=[dense1_output_size],
                                      initializer=tf.constant_initializer(0.1))

      # for testing
      dense1_W_penalty = tf.multiply(tf.nn.l2_loss(dense1_weights),
                                     deepnovo_config.l2_loss_weight,
                                     name='dense1_W_penalty')

      # dense2: # [128, 512] >> [128, 512]
      #~ dense2_input_size = 512
      #~ dense2_output_size = 512
      #~ dense2_weights = tf.get_variable(
          #~ "dense2_weights",
          #~ shape=[dense2_input_size, dense2_output_size],
          #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
      #~ dense2_biases = tf.get_variable("dense2_biases",
                                      #~ shape=[dense2_output_size],
                                      #~ initializer=tf.constant_initializer(0.1))

      # logit_linear: [128, 512] >> [128, 27]
      #~ linear_input_size = 512
      #~ linear_output_size = deepnovo_config.vocab_size
      #~ linear_weights = tf.get_variable(
          #~ "linear_weights",
          #~ shape=[linear_input_size, linear_output_size])
      #~ linear_biases = tf.get_variable("linear_biases",
                                      #~ shape=[linear_output_size],
                                      #~ initializer=tf.constant_initializer(0.0))

    # LSTM-Intensity Connection-Model Parameters
    #~ denseL_W = variable_scope.get_variable(
        #~ name="denseL_W",
        #~ shape=[deepnovo_config.vocab_size, deepnovo_config.vocab_size],
        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ denseI_W = variable_scope.get_variable(
        #~ name="denseI_W",
        #~ shape=[deepnovo_config.vocab_size, deepnovo_config.vocab_size],
        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ denseC_B = variable_scope.get_variable(
        #~ name="denseC_B",
        #~ shape=[deepnovo_config.vocab_size],
        #~ initializer=tf.constant_initializer(0.1))
    # cat
    dense_concat_W = variable_scope.get_variable(
        name="dense_concat_W",
        shape=[512 + 512, 512],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense_concat_B = variable_scope.get_variable(
        name="dense_concat_B",
        shape=[512],
        initializer=tf.constant_initializer(0.1))

    # DECODING - SPECTRUM as Input 0
    with variable_scope.variable_scope("LSTM_cell"):
      input0 = encoded_spectrum
      batch_size = array_ops.shape(input0)[0]
      zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      #~ _, lstm_state = cell(inputs=input0,state=zero_state)
      # nobi
      _, lstm_state_0 = cell(inputs=input0, state=zero_state)

    # nobi
    # DECODING - lstm_input_projected
    with variable_scope.variable_scope("LSTM_input_projected"):
      lstm_input_projected_W = variable_scope.get_variable(
          name="lstm_input_projected_W",
          shape=[deepnovo_config.embedding_size, deepnovo_config.num_units])
      lstm_input_projected_B = variable_scope.get_variable(
          name="lstm_input_projected_B",
          shape=[deepnovo_config.num_units],
          initializer=tf.constant_initializer(0.1))

    # DECODING LOOP
    # nobi
    outputs = []
    AA_1 = decoder_inputs_emb[0] # padding [AA_1, AA_2, ?] with GO/EOS
    # ltsm.len_full
    lstm_state = lstm_state_0

    for i, AA_2 in enumerate(decoder_inputs_emb):

      # nobi
      if i > 0: # to-do-later: bring variable definitions out of the loop
        variable_scope.get_variable_scope().reuse_variables()

      # INTENSITY-Model
      candidate_intensity = intensity_inputs[i] # [128, 27, 2, 10]

      if deepnovo_config.FLAGS.shared: # shared-weight

        candidate_intensity_reshape = tf.reshape(candidate_intensity,
                                                 shape=[-1, dense1_input_size]) # [128*27, 2*10]

        layer_dense1_input = candidate_intensity_reshape
        layer_dense1 = tf.nn.relu(tf.matmul(layer_dense1_input, dense1_W)
                                  + dense1_B) # [128*27, 1024]
        layer_dense1_drop = tf.nn.dropout(layer_dense1, keep_dense)
        layer_dense1_output = (tf.matmul(layer_dense1_drop, dense_linear_W)
                               + dense_linear_B) # [128*27,1]

        # Intensity output
        intensity_output = tf.reshape(layer_dense1_output,
                                      shape=[-1, deepnovo_config.vocab_size]) # [128,27]

      else: # joint-weight

        # image_batch: [128, 26, 8, 20] >> [128, 8, 20, 26]
        # This is a bug, should be fixed at the input processing later.
        image_batch = tf.transpose(candidate_intensity, perm=[0, 2, 3, 1]) # [128,8,20,26]

        # conv1: [128, 8, 20, 26] >> [128, 8, 20, 64] with kernel [1, 3, 26, 64]
        conv1 = tf.nn.relu(tf.nn.conv2d(image_batch,
                                        conv1_weights,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
                           + conv1_biases)

        # conv2: [128, 8, 20, 64] >> [128, 8, 20, 64] with kernel [1, 2, 64, 64]
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                        conv2_weights,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
                           + conv2_biases)
        conv2 = tf.nn.max_pool(conv2,
                               ksize=[1, 1, 3, 1],
                               strides=[1, 1, 2, 1],
                               padding='SAME') # [128, 8, 10, 64]
        conv2 = tf.nn.dropout(conv2, keep_conv)

        # dense1: 4D >> [128, 512]
        dense1_input = tf.reshape(conv2, [-1, dense1_input_size]) # 2D flatten
        dense1 = tf.nn.relu(tf.matmul(dense1_input, dense1_weights)
                            + dense1_biases) # [128, 512]

        # dense2: # [128, 512] >> [128, 512]
        #~ dense2 = tf.nn.relu(tf.matmul(dense1, dense2_weights) + dense2_biases) # [128, 512]

        #~ dropout1 = tf.nn.dropout(dense2, keep_dense, name="dropout1")
        dropout1 = tf.nn.dropout(dense1, keep_dense, name="dropout1")

        # logit_linear: [128, 512] >> [128, 27]
        #~ intensity_output = tf.add(tf.matmul(dropout1, linear_weights),
                                  #~ linear_biases) # [128, 27]
        intensity_output = dropout1
        with variable_scope.variable_scope("intensity_output_projected"):
          intensity_output_projected = rnn_cell_impl._linear( # TODO(nh2tran): _linear
              args=intensity_output,
              output_size=deepnovo_config.vocab_size, # [128,27]
              bias=True,
              bias_initializer=None,#0.1,
              kernel_initializer=None)

      # nobi
      # LSTM-Model
      AA_1_projected = (tf.matmul(AA_1, lstm_input_projected_W)
                        + lstm_input_projected_B)
      AA_2_projected = (tf.matmul(AA_2, lstm_input_projected_W)
                        + lstm_input_projected_B)

      with variable_scope.variable_scope("LSTM_cell"):

        variable_scope.get_variable_scope().reuse_variables()

        # nobi
        #~ _, lstm_state_1 = cell(inputs=AA_1_projected, state=lstm_state_0)
        #~ lstm_output, _ = cell(inputs=AA_2_projected, state=lstm_state_1)
        #
        # lstm.len_full
        lstm_output, lstm_state = cell(inputs=AA_2_projected, state=lstm_state)

        AA_1 = AA_2

      with variable_scope.variable_scope("lstm_output_projected"):
        lstm_output_projected = rnn_cell_impl._linear( # TODO(nh2tran): _linear
            args=lstm_output,
            output_size=deepnovo_config.vocab_size, # [128,27]
            bias=True,
            bias_initializer=None,#0.1,
            kernel_initializer=None)

      # LSTM-Intensity Connection-Model >> OUTPUT
      if deepnovo_config.FLAGS.use_intensity and deepnovo_config.FLAGS.use_lstm:

        #~ output_logit = tf.nn.relu(tf.matmul(lstm_output_projected, denseL_W)
                                  #~ + tf.matmul(intensity_output_projected, denseI_W)
                                  #~ + denseC_B)

        # cat
        concat = tf.concat(axis=1, values=[intensity_output, lstm_output])
        concat_dense = tf.nn.relu(tf.matmul(concat, dense_concat_W)
                                  + dense_concat_B)
        concat_drop = tf.nn.dropout(concat_dense, keep_dense)

        with variable_scope.variable_scope("output_logit"):
          output_logit = rnn_cell_impl._linear(args=concat_drop, # TODO(nh2tran): _linear
                                               output_size=deepnovo_config.vocab_size, # [128,27]
                                               bias=True,
                                               bias_initializer=None,#0.1,
                                               kernel_initializer=None)

      elif deepnovo_config.FLAGS.use_intensity:
        # intensity only (without LSTM >> up to 10% loss, especially at AA-accuracy?)
        output_logit = intensity_output_projected

      elif deepnovo_config.FLAGS.use_lstm:
        output_logit = lstm_output_projected

      else:
        print("ERROR: wrong LSTM-Intensity model specified!")
        sys.exit()

      outputs.append(output_logit)

  return (outputs, dense1_W_penalty)


def embed_labels(encoded_spectrum,
                 intensity_inputs_forward,
                 intensity_inputs_backward,
                 decoder_inputs_forward,
                 decoder_inputs_backward,
                 keep_conv,
                 keep_dense):
  """TODO(nh2tran): docstring."""

  with variable_scope.variable_scope("embedding_rnn_decoder"):
    with ops.device("/cpu:0"):
      embedding = variable_scope.get_variable(
          name="embedding",
          shape=[deepnovo_config.vocab_size, deepnovo_config.embedding_size])

    # nobi
    decoder_inputs_forward_emb = [embedding_ops.embedding_lookup(embedding, x)
                                  for x in decoder_inputs_forward]
    decoder_inputs_backward_emb = [embedding_ops.embedding_lookup(embedding, x)
                                   for x in decoder_inputs_backward]

    return (decode_spectrum(encoded_spectrum,
                            intensity_inputs_forward,
                            decoder_inputs_forward_emb,
                            keep_conv,
                            keep_dense,
                            scope="rnn_decoder_forward"),
            decode_spectrum(encoded_spectrum,
                            intensity_inputs_backward,
                            decoder_inputs_backward_emb,
                            keep_conv,
                            keep_dense,
                            scope="rnn_decoder_backward"))


def encode_spectrum(encoder_inputs,
                    intensity_inputs_forward,
                    intensity_inputs_backward,
                    decoder_inputs_forward,
                    decoder_inputs_backward,
                    keep_conv,
                    keep_dense):
  """TODO(nh2tran): docstring."""

  with variable_scope.variable_scope("embedding_rnn_seq2seq"):

    # spectra_holder
    layer0 = tf.reshape(encoder_inputs[0], [-1, 1, deepnovo_config.MZ_SIZE, 1])

    # conv1
    conv1_W = variable_scope.get_variable(
        name="conv1_W",
        shape=[1, 4, 1, 4],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv1_B = variable_scope.get_variable(
        name="conv1_B",
        shape=[4],
        initializer=tf.constant_initializer(0.1))

    # conv2
    conv2_W = variable_scope.get_variable(
        name="conv2_W",
        shape=[1, 4, 4, 4],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv2_B = variable_scope.get_variable(
        name="conv2_B",
        shape=[4],
        initializer=tf.constant_initializer(0.1))

    # pool1 [1, 1, 4, 1]

    # conv3
    #~ conv3_W = variable_scope.get_variable(
        #~ name="conv3_W",
        #~ shape=[1, 4, 4, 4],
        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ conv3_B = variable_scope.get_variable(
        #~ name="conv3_B",
        #~ shape=[4],
        #~ initializer=tf.constant_initializer(0.1))

    # pool2 [1, 1, 4, 1]

    # dense1
    dense1_input_size = 1 * (deepnovo_config.MZ_SIZE // (4)) * 4
    dense1_output_size = 512
    dense1_W = variable_scope.get_variable(
        name="dense1_W",
        shape=[dense1_input_size, dense1_output_size],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense1_B = variable_scope.get_variable(
        name="dense1_B",
        shape=[dense1_output_size],
        initializer=tf.constant_initializer(0.1))

    # dense2
    #~ dense2_input_size = dense1_output_size
    #~ dense2_output_size = 512
    #~ dense2_W = variable_scope.get_variable(
        #~ name="dense2_W",
        #~ shape=[dense2_input_size, dense2_output_size],
        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ dense2_B = variable_scope.get_variable(
        #~ name="dense2_B",
        #~ shape=[dense2_output_size],
        #~ initializer=tf.constant_initializer(0.1))

    # layers
    conv1 = tf.nn.relu(tf.nn.conv2d(layer0,
                                    conv1_W,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                       + conv1_B)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                    conv2_W,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                       + conv2_B)
    conv2 = tf.nn.max_pool(conv2,
                           ksize=[1, 1, 6, 1],
                           strides=[1, 1, 4, 1],
                           padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_conv)

    #~ conv3 = tf.nn.relu(tf.nn.conv2d(conv2,
                                    #~ conv3_W,
                                    #~ strides=[1, 1, 1, 1],
                                    #~ padding='SAME')
                       #~ + conv3_B)
    #~ conv3 = tf.nn.max_pool(conv3,
                           #~ ksize=[1, 1, 6, 1],
                           #~ strides=[1, 1, 4, 1],
                           #~ padding='SAME')
    #~ conv3 = tf.nn.dropout(conv3, keep_conv)

    dense1 = tf.reshape(conv2, [-1, dense1_input_size])
    dense1 = tf.nn.relu(tf.matmul(dense1, dense1_W) + dense1_B)
    dense1 = tf.nn.dropout(dense1, keep_dense)

    #~ dense2 = tf.nn.relu(tf.matmul(dense1, dense2_W) + dense2_B)
    #~ dense2 = tf.nn.dropout(dense2, keep_dense)

    # SPECTRUM as Input 0
    encoded_spectrum = dense1
    #~ encoded_spectrum = tf.zeros(shape=array_ops.shape(layer_dense1_drop))

    return embed_labels(encoded_spectrum,
                        intensity_inputs_forward,
                        intensity_inputs_backward,
                        decoder_inputs_forward,
                        decoder_inputs_backward,
                        keep_conv,
                        keep_dense)


def train(encoder_inputs,
          intensity_inputs_forward,
          intensity_inputs_backward,
          decoder_inputs_forward,
          decoder_inputs_backward,
          targets_forward,
          targets_backward,
          target_weights,
          keep_conv,
          keep_dense):
  """TODO(nh2tran): docstring."""

  all_inputs = (encoder_inputs
                + intensity_inputs_forward
                + intensity_inputs_backward
                + decoder_inputs_forward
                + decoder_inputs_backward
                + targets_forward
                + targets_backward
                + target_weights)

  losses = []

  outputs_forward = []
  outputs_backward = []

  #~ with tf.name_scope(name="model_with_buckets", values=all_inputs):
  with ops.op_scope(all_inputs, name="model_with_buckets"):

    for j, bucket in enumerate(deepnovo_config._buckets): # TODO(nh2tran): _buckets

      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):

        # for testing
        #~ bucket_outputs_forward, bucket_outputs_backward = encode_spectrum(
        ((bucket_outputs_forward,
          penalty_forward),
         (bucket_outputs_backward,
          penalty_backward)) = encode_spectrum(encoder_inputs,
                                               intensity_inputs_forward[:bucket],
                                               intensity_inputs_backward[:bucket],
                                               decoder_inputs_forward[:bucket],
                                               decoder_inputs_backward[:bucket],
                                               keep_conv,
                                               keep_dense)

        outputs_forward.append(bucket_outputs_forward)
        outputs_backward.append(bucket_outputs_backward)

        # losses depend on direction
        if deepnovo_config.FLAGS.direction == 0:

          losses.append(sequence_loss(outputs_forward[-1],
                                      targets_forward[:bucket],
                                      target_weights[:bucket],
                                      name="sequence_loss_forward"))

        elif deepnovo_config.FLAGS.direction == 1:

          losses.append(sequence_loss(outputs_backward[-1],
                                      targets_backward[:bucket],
                                      target_weights[:bucket],
                                      name="sequence_loss_backward"))

        else:

          losses.append((sequence_loss(outputs_forward[-1],
                                       targets_forward[:bucket],
                                       target_weights[:bucket],
                                       name="sequence_loss_forward")
                         + sequence_loss(outputs_backward[-1],
                                         targets_backward[:bucket],
                                         target_weights[:bucket],
                                         name="sequence_loss_backward")) / 2)

          # for testing
          losses[-1] += penalty_forward + penalty_backward

  return outputs_forward, outputs_backward, losses
