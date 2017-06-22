from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import data_utils


def sequence_loss_per_sample(logits,
                             targets, 
                             weights):
  #~ print("sequence_loss_per_sample()")

  #~ with tf.name_scope(name="sequence_loss_by_example", values=logits + targets + weights):
  with ops.op_scope(logits + targets + weights, None, "sequence_loss_by_example"):

    log_perp_list = []
    
    for logit, target, weight in zip(logits, targets, weights):
      target = array_ops.reshape(math_ops.to_int64(target), [-1])
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logit, target)
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
  #~ print("sequence_loss()")

  #~ with tf.name_scope(name=name, values=logits + targets + weights):
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
  #~ print("decode_spectrum()")

  single_cell = rnn_cell.BasicLSTMCell(num_units=data_utils.num_units, state_is_tuple=True)
  #~ single_cell = rnn_cell.BasicRNNCell(num_units=data_utils.num_units)
  #~ single_cell = rnn_cell.GRUCell(num_units=data_utils.num_units)
  if (data_utils.num_layers > 1):
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * data_utils.num_layers)
  else:
    cell = single_cell
  cell = rnn_cell.DropoutWrapper(cell, 
                                 input_keep_prob=keep_dense, 
                                 output_keep_prob=keep_dense)
  
  with variable_scope.variable_scope(scope):

    # INTENSITY-Model Parameters
    # intensity input [128,27,2,10]
    #
    if (data_utils.FLAGS.shared): # shared-weight
      dense1_input_size = data_utils.num_ion * data_utils.WINDOW_SIZE
      dense1_output_size = 1024
      #
      dense1_W = variable_scope.get_variable(
                                name="dense1_W_0", 
                                shape=[dense1_input_size, dense1_output_size],
                                initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_B = variable_scope.get_variable(
                                name="dense1_B_0", 
                                shape=[dense1_output_size],
                                initializer=tf.constant_initializer(0.1))
      #
      dense_linear_W = variable_scope.get_variable(
                                      name="dense_linear_W", 
                                      shape=[dense1_output_size, 1])
      #
      dense_linear_B = variable_scope.get_variable(
                                      name="dense_linear_B", 
                                      shape=[1],
                                      initializer=tf.constant_initializer(0.1))
    #   
    else: # joint-weight

      # conv1: [128,8,20,26] >> [128,8,20,64] with kernel [1,3,26,64]
      conv1_weights = tf.get_variable(name="conv1_weights", 
                                      shape=[1,3,data_utils.vocab_size,64],
                                      initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv1_biases = tf.get_variable(name="conv1_biases", 
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.1))

      # conv2: [128,8,20,64] >> [128,8,20,64] with kernel [1,2,64,64]
      conv2_weights = tf.get_variable(name="conv2_weights", 
                                      shape=[1,2,64,64],
                                      initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv2_biases = tf.get_variable(name="conv2_biases", 
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.1))
                                     
      # max_pool: [128,8,20,64] >> [128,8,10,64]

      # dense1: # 4D >> [128,512]
      dense1_input_size = data_utils.num_ion * (data_utils.WINDOW_SIZE // 2) * 64 # data_utils.vocab_size
      dense1_output_size = 512
      dense1_weights = tf.get_variable("dense1_weights", 
                                        shape=[dense1_input_size, dense1_output_size],
                                        initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_biases = tf.get_variable("dense1_biases", shape=[dense1_output_size], initializer=tf.constant_initializer(0.1))
      #
      # for testing
      dense1_W_penalty = tf.mul(tf.nn.l2_loss(dense1_weights), data_utils.l2_loss_weight, name='dense1_W_penalty')

      # dense2: # [128,512] >> [128,512]
      #~ dense2_input_size = 512
      #~ dense2_output_size = 512
      #~ dense2_weights = tf.get_variable("dense2_weights", 
                                        #~ shape=[dense2_input_size, dense2_output_size],
                                        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
      #~ dense2_biases = tf.get_variable("dense2_biases", shape=[dense2_output_size], initializer=tf.constant_initializer(0.1))
      
      # logit_linear: [128,512] >> [128,27]
      #~ linear_input_size = 512
      #~ linear_output_size = data_utils.vocab_size
      #~ linear_weights = tf.get_variable("linear_weights", 
                                        #~ shape=[linear_input_size, linear_output_size])
      #~ linear_biases = tf.get_variable("linear_biases", shape=[linear_output_size], initializer=tf.constant_initializer(0.0))






    # LSTM-Intensity Connection-Model Parameters
    #
    #~ denseL_W = variable_scope.get_variable(name="denseL_W",shape=[data_utils.vocab_size,data_utils.vocab_size],
                                           #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ denseI_W = variable_scope.get_variable(name="denseI_W",shape=[data_utils.vocab_size,data_utils.vocab_size],
                                           #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ denseC_B = variable_scope.get_variable(name="denseC_B",shape=[data_utils.vocab_size],
                                           #~ initializer=tf.constant_initializer(0.1))
    # cat
    dense_concat_W = variable_scope.get_variable(name="dense_concat_W",shape=[512+512, 512],
                                           initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense_concat_B = variable_scope.get_variable(name="dense_concat_B",shape=[512],
                                           initializer=tf.constant_initializer(0.1))






    # DECODING - SPECTRUM as Input 0
    with variable_scope.variable_scope("LSTM_cell"):
      #
      input0 = encoded_spectrum
      #
      batch_size = array_ops.shape(input0)[0]
      zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      #
      #~ _, lstm_state = cell(inputs=input0,state=zero_state)
      # nobi
      _, lstm_state_0 = cell(inputs=input0, state=zero_state)
      





    # nobi
    # DECODING - lstm_input_projected
    with variable_scope.variable_scope("LSTM_input_projected"):
      lstm_input_projected_W = variable_scope.get_variable(name="lstm_input_projected_W",
                                                           shape=[data_utils.embedding_size,data_utils.num_units])
      #
      lstm_input_projected_B = variable_scope.get_variable(name="lstm_input_projected_B",
                                                           shape=[data_utils.num_units],
                                                           initializer=tf.constant_initializer(0.1))
      





    # DECODING LOOP
    # nobi
    outputs = []
    AA_1 = decoder_inputs_emb[0] # padding [AA_1, AA_2, ?] with GO/EOS
    for i, AA_2 in enumerate(decoder_inputs_emb):
      
      # nobi
      if (i > 0): # to-do-later: bring variable definitions out of the loop
        variable_scope.get_variable_scope().reuse_variables()






      # INTENSITY-Model
      candidate_intensity = intensity_inputs[i] # [128,27,2,10]
      #
      if (data_utils.FLAGS.shared): # shared-weight
        candidate_intensity_reshape = tf.reshape(candidate_intensity,
                                                 shape=[-1,dense1_input_size]) # [128*27,2*10]
        #
        layer_dense1_input = candidate_intensity_reshape
        #
        layer_dense1 = tf.nn.relu(tf.matmul(layer_dense1_input,dense1_W)+dense1_B) # [128*27,1024]
        #
        layer_dense1_drop = tf.nn.dropout(layer_dense1,keep_dense)
        #
        layer_dense1_output = tf.matmul(layer_dense1_drop,dense_linear_W)+dense_linear_B # [128*27,1]
        #
        # Intensity output
        intensity_output = tf.reshape(layer_dense1_output,
                                      shape=[-1,data_utils.vocab_size]) # [128,27]
      #   
      else: # joint-weight

        # image_batch: [128,26,8,20] >> [128,8,20,26]
        # This is a bug, should be fixed at the input processing later.
        image_batch = tf.transpose(candidate_intensity, perm=[0,2,3,1])  # [128,8,20,26]

        # conv1: [128,8,20,26] >> [128,8,20,64] with kernel [1,3,26,64]
        conv1 = tf.nn.relu(tf.nn.conv2d(image_batch, conv1_weights, strides=[1,1,1,1], padding='SAME') + conv1_biases)

        # conv2: [128,8,20,64] >> [128,8,20,64] with kernel [1,2,64,64]
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_weights, strides=[1,1,1,1], padding='SAME') + conv2_biases)
        conv2 = tf.nn.max_pool(conv2, ksize=[1,1,3,1],strides=[1,1,2,1], padding='SAME') # [128,8,10,64]
        conv2 = tf.nn.dropout(conv2, keep_conv)

        # dense1: 4D >> [128,512]
        dense1_input = tf.reshape(conv2, [-1, dense1_input_size]) # 2D flatten
        dense1 = tf.nn.relu(tf.matmul(dense1_input, dense1_weights) + dense1_biases) # [128,512]

        # dense2: # [128,512] >> [128,512]
        #~ dense2 = tf.nn.relu(tf.matmul(dense1, dense2_weights) + dense2_biases) # [128,512]

        #~ dropout1 = tf.nn.dropout(dense2, keep_dense, name="dropout1")
        dropout1 = tf.nn.dropout(dense1, keep_dense, name="dropout1")

        # logit_linear: [128,512] >> [128,27]
        #~ intensity_output = tf.add(tf.matmul(dropout1, linear_weights), linear_biases) # [128,27]
        intensity_output = dropout1
        intensity_output_projected = rnn_cell._linear(intensity_output, data_utils.vocab_size, # [128,27]
                                                      bias=True,
                                                      bias_start=0.1,
                                                      scope="intensity_output_projected") 






      # nobi
      # LSTM-Model
      AA_1_projected = tf.matmul(AA_1, lstm_input_projected_W) + lstm_input_projected_B
      AA_2_projected = tf.matmul(AA_2, lstm_input_projected_W) + lstm_input_projected_B
      #
      with variable_scope.variable_scope("LSTM_cell"):
        #
        variable_scope.get_variable_scope().reuse_variables()
        #
        _, lstm_state_1 = cell(inputs=AA_1_projected, state=lstm_state_0)
        lstm_output, _ = cell(inputs=AA_2_projected, state=lstm_state_1)
        #
        AA_1 = AA_2
      #
      lstm_output_projected = rnn_cell._linear(lstm_output,data_utils.vocab_size, # [128,27]
                                              bias=True,
                                              bias_start=0.1,
                                              scope="lstm_output_projected")
      





      # LSTM-Intensity Connection-Model >> OUTPUT
      #
      if (data_utils.FLAGS.use_intensity and data_utils.FLAGS.use_lstm):
        #
        #~ output_logit = tf.nn.relu(tf.matmul(lstm_output_projected,denseL_W) +
                                  #~ tf.matmul(intensity_output_projected,denseI_W) + 
                                  #~ denseC_B)
        #
        # cat
        concat = tf.concat(concat_dim=1, values=[intensity_output, lstm_output])
        concat_dense = tf.nn.relu(tf.matmul(concat, dense_concat_W) + dense_concat_B)
        concat_drop = tf.nn.dropout(concat_dense, keep_dense)
        #
        output_logit = rnn_cell._linear(concat_drop, data_utils.vocab_size, # [128,27]
                                        bias=True,
                                        bias_start=0.1,
                                        scope="concat_output_projected")
      #
      elif (data_utils.FLAGS.use_intensity):
        # intensity only (without LSTM >> up to 10% loss, especially at AA-accuracy?)
        output_logit = intensity_output_projected
      #
      elif (data_utils.FLAGS.use_lstm):
        output_logit = lstm_output_projected
      #
      else:
        print("ERROR: wrong LSTM-Intensity model specified!")
        sys.exit()
      #
      outputs.append(output_logit)


  return (outputs, dense1_W_penalty)


def embed_labels(encoded_spectrum,
                 intensity_inputs_forward,
                 intensity_inputs_backward,
                 decoder_inputs_forward, 
                 decoder_inputs_backward,
                 keep_conv, 
                 keep_dense):
  #~ print("embed_labels()")

  with variable_scope.variable_scope("embedding_rnn_decoder"):
    with ops.device("/cpu:0"):
      embedding = variable_scope.get_variable(name="embedding", 
                                              shape=[data_utils.vocab_size,data_utils.embedding_size])

    # nobi
    decoder_inputs_forward_emb = [embedding_ops.embedding_lookup(embedding, x) for x in decoder_inputs_forward]
    decoder_inputs_backward_emb = [embedding_ops.embedding_lookup(embedding, x) for x in decoder_inputs_backward]

    return (decode_spectrum(encoded_spectrum,
                            intensity_inputs_forward,
                            decoder_inputs_forward_emb,
                            keep_conv, 
                            keep_dense,
                            scope="rnn_decoder_forward"),
           #
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
  #~ print("encode_spectrum()")

  with variable_scope.variable_scope("embedding_rnn_seq2seq"):
    
    # spectra_holder
    layer0 = tf.reshape(encoder_inputs[0], [-1,1,data_utils.MZ_SIZE,1])

    # conv1
    conv1_W = variable_scope.get_variable(name="conv1_W", shape=[1,4,1,4],
                                        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv1_B = variable_scope.get_variable(name="conv1_B", shape=[4],
                                        initializer=tf.constant_initializer(0.1))
    #
    # conv2
    conv2_W = variable_scope.get_variable(name="conv2_W", shape=[1,4,4,4],
                                        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv2_B = variable_scope.get_variable(name="conv2_B", shape=[4],
                                        initializer=tf.constant_initializer(0.1))
    #
    # pool1 [1,1,4,1]
    #
    #~ # conv3
    #~ conv3_W = variable_scope.get_variable(name="conv3_W", shape=[1,4,4,4],
                                        #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ conv3_B = variable_scope.get_variable(name="conv3_B", shape=[4],
                                        #~ initializer=tf.constant_initializer(0.1))
    #~ #
    #~ # pool2 [1,1,4,1]
    #
    # dense1
    dense1_input_size = 1 * (data_utils.MZ_SIZE // (4)) * 4
    dense1_output_size = 512
    dense1_W = variable_scope.get_variable(name="dense1_W", shape=[dense1_input_size, dense1_output_size],
                                           initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense1_B = variable_scope.get_variable(name="dense1_B", shape=[dense1_output_size],
                                           initializer=tf.constant_initializer(0.1))
    #
    # dense2
    #~ dense2_input_size = dense1_output_size
    #~ dense2_output_size = 512
    #~ dense2_W = variable_scope.get_variable(name="dense2_W", shape=[dense2_input_size, dense2_output_size],
                                           #~ initializer=tf.uniform_unit_scaling_initializer(1.43))
    #~ dense2_B = variable_scope.get_variable(name="dense2_B", shape=[dense2_output_size],
                                           #~ initializer=tf.constant_initializer(0.1))

    # layers
    conv1 = tf.nn.relu(tf.nn.conv2d(layer0, conv1_W, strides=[1,1,1,1], padding='SAME') + conv1_B)
    #
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding='SAME') + conv2_B)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,1,6,1], strides=[1,1,4,1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_conv)
    #
    #~ conv3 = tf.nn.relu(tf.nn.conv2d(conv2, conv3_W, strides=[1,1,1,1], padding='SAME') + conv3_B)
    #~ conv3 = tf.nn.max_pool(conv3, ksize=[1,1,6,1], strides=[1,1,4,1], padding='SAME')
    #~ conv3 = tf.nn.dropout(conv3, keep_conv)
    #
    dense1 = tf.reshape(conv2, [-1, dense1_input_size])
    dense1 = tf.nn.relu(tf.matmul(dense1, dense1_W) + dense1_B)
    dense1 = tf.nn.dropout(dense1, keep_dense)
    #
    #~ dense2 = tf.nn.relu(tf.matmul(dense1, dense2_W) + dense2_B)
    #~ dense2 = tf.nn.dropout(dense2, keep_dense)

    # SPECTRUM as Input 0
    #
    encoded_spectrum = dense1
    #~ #
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
  #~ print("train()")

  all_inputs = (encoder_inputs + 
                intensity_inputs_forward + 
                intensity_inputs_backward + 
                decoder_inputs_forward + 
                decoder_inputs_backward + 
                targets_forward + 
                targets_backward + 
                target_weights)
                
  losses = []
  
  outputs_forward = []
  outputs_backward = []
  
  #~ with tf.name_scope(name="model_with_buckets", values=all_inputs):
  with ops.op_scope(all_inputs, name="model_with_buckets"):
    #
    for j, bucket in enumerate(data_utils._buckets):
      #
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
                                           
        # for testing
        #~ bucket_outputs_forward, bucket_outputs_backward = encode_spectrum(
        ((bucket_outputs_forward, penalty_forward), (bucket_outputs_backward, penalty_backward)) = encode_spectrum(
                                                            encoder_inputs,
                                                            intensity_inputs_forward[:bucket],
                                                            intensity_inputs_backward[:bucket],
                                                            decoder_inputs_forward[:bucket],
                                                            decoder_inputs_backward[:bucket],
                                                            keep_conv,
                                                            keep_dense)

        outputs_forward.append(bucket_outputs_forward)
        outputs_backward.append(bucket_outputs_backward)
        
        # losses depend on direction
        if data_utils.FLAGS.direction==0:
          losses.append(sequence_loss(outputs_forward[-1], 
                                      targets_forward[:bucket],
                                      target_weights[:bucket],
                                      name="sequence_loss_forward"))
        elif data_utils.FLAGS.direction==1:
          losses.append(sequence_loss(outputs_backward[-1], 
                                      targets_backward[:bucket],
                                      target_weights[:bucket],
                                      name="sequence_loss_backward"))
        else:
          losses.append((sequence_loss(outputs_forward[-1], 
                                       targets_forward[:bucket],
                                       target_weights[:bucket],
                                       name="sequence_loss_forward") + 
                         #
                         sequence_loss(outputs_backward[-1], 
                                       targets_backward[:bucket],
                                       target_weights[:bucket],
                                       name="sequence_loss_backward")) / 2)
          
          # for testing
          losses[-1] += penalty_forward + penalty_backward
  
  return outputs_forward, outputs_backward, losses
