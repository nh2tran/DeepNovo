from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#~ from tensorflow.models.rnn import rnn_cell

#~ from tensorflow.models.rnn import seq2seq
import novonet_training
import novonet_decoding
#~ from tensorflow.models.rnn.translate import data_utils
import data_utils


class TrainingModel(object):

  def __init__(self,
               session,
               training_mode):

    #~ print("Seq2SeqModel: __init__()")
    
    self.global_step = tf.Variable(0,trainable=False)
    
    # Dropout probabilities
    self.keep_conv_holder = tf.placeholder(dtype=tf.float32,name="keep_conv")
    self.keep_dense_holder = tf.placeholder(dtype=tf.float32,name="keep_dense")

    
    
    
    
    
    # INPUT PLACEHOLDERS

    # spectrum
    self.encoder_inputs = [tf.placeholder(dtype=tf.float32, 
                                          shape=[None,data_utils.MZ_SIZE], 
                                          name="encoder_inputs")]

    # candidate intensity
    self.intensity_inputs_forward = []
    self.intensity_inputs_backward = []
    #
    for x in xrange(data_utils._buckets[-1]):
      #
      self.intensity_inputs_forward.append(tf.placeholder(dtype=tf.float32,
                                                          shape=[None,data_utils.vocab_size,data_utils.num_ion,data_utils.WINDOW_SIZE],
                                                          name="intensity_inputs_forward_{0}".format(x)))
      #
      self.intensity_inputs_backward.append(tf.placeholder(tf.float32,
                                                            shape=[None,data_utils.vocab_size,data_utils.num_ion,data_utils.WINDOW_SIZE],
                                                            name="intensity_inputs_backward_{0}".format(x)))

    # decoder inputs
    self.decoder_inputs_forward = []
    self.decoder_inputs_backward = []
    #
    self.target_weights = []
    #
    for x in xrange(data_utils._buckets[-1] + 1):
      #
      self.decoder_inputs_forward.append(tf.placeholder(dtype=tf.int32,
                                                        shape=[None],
                                                        name="decoder_inputs_forward_{0}".format(x)))
      #
      self.decoder_inputs_backward.append(tf.placeholder(dtype=tf.int32,
                                                         shape=[None],
                                                         name="decoder_inputs_backward_{0}".format(x)))
      #
      self.target_weights.append(tf.placeholder(dtype=tf.float32,
                                                shape=[None],
                                                name="target_weights_{0}".format(x)))

    # Our targets are decoder inputs shifted by one.
    self.targets_forward = [self.decoder_inputs_forward[x + 1]
                            for x in xrange(len(self.decoder_inputs_forward) - 1)]
    #
    self.targets_backward = [self.decoder_inputs_backward[x + 1]
                             for x in xrange(len(self.decoder_inputs_backward) - 1)]






    # OUTPUTS and LOSSES
    self.outputs_forward, self.outputs_backward, self.losses = novonet_training.train(
    #
                          self.encoder_inputs, 
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
      for b in xrange(len(data_utils._buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, data_utils.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

      # for TensorBoard
      #~ self.train_writer = tf.train.SummaryWriter(data_utils.FLAGS.train_dir, session.graph)
      #~ self.loss_summaries = [tf.scalar_summary("losses_" + str(b), self.losses[b])
                             #~ for b in xrange(len(data_utils._buckets))]
      #~ dense1_W_penalty = tf.get_default_graph().get_tensor_by_name(
                         #~ "model_with_buckets/embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/dense1_W_penalty:0")
      #~ self.dense1_W_penalty_summary = tf.scalar_summary("dense1_W_penalty_summary", dense1_W_penalty)

    # Saver
    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    

  def step(self, session, encoder_inputs,
           intensity_inputs_forward=None,
           intensity_inputs_backward=None,
           decoder_inputs_forward=None, 
           decoder_inputs_backward=None, 
           target_weights=None,
           bucket_id=0, 
           training_mode=True):
    #~ print("step()")

    # Check if the sizes match.
    decoder_size = data_utils._buckets[bucket_id]

    # Input feed
    input_feed = {}
    #
    input_feed[self.encoder_inputs[0].name] = encoder_inputs[0]
    #
    if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==2):
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_forward[x].name] = intensity_inputs_forward[x]
        input_feed[self.decoder_inputs_forward[x].name] = decoder_inputs_forward[x]
      # Since our targets are decoder inputs shifted by one, we need one more.
      last_target_forward = self.decoder_inputs_forward[decoder_size].name
      input_feed[last_target_forward] = np.zeros([encoder_inputs[0].shape[0]],dtype=np.int32)
    #
    if (data_utils.FLAGS.direction==1 or data_utils.FLAGS.direction==2):
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_backward[x].name] = intensity_inputs_backward[x]
        input_feed[self.decoder_inputs_backward[x].name] = decoder_inputs_backward[x]
      #
      last_target_backward = self.decoder_inputs_backward[decoder_size].name
      input_feed[last_target_backward] = np.zeros([encoder_inputs[0].shape[0]],dtype=np.int32)
    #
    for x in xrange(decoder_size):
      input_feed[self.target_weights[x].name] = target_weights[x]

    
    # keeping probability for dropout layers
    if training_mode:
      input_feed[self.keep_conv_holder.name] = data_utils.keep_conv
      input_feed[self.keep_dense_holder.name] = data_utils.keep_dense
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
    if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==2):
      for x in xrange(decoder_size):  
        output_feed.append(self.outputs_forward[bucket_id][x])

    # Output backward logits
    if (data_utils.FLAGS.direction==1 or data_utils.FLAGS.direction==2):
      for x in xrange(decoder_size):  
        output_feed.append(self.outputs_backward[bucket_id][x])

    # RUN
    outputs = session.run(fetches=output_feed, feed_dict=input_feed)
    #
    # for TensorBoard
    #~ if (training_mode and (self.global_step.eval() % data_utils.steps_per_checkpoint == 0)):
      #~ summary_op = tf.merge_summary([self.loss_summaries[bucket_id], self.dense1_W_penalty_summary])
      #~ summary_str = session.run(summary_op, feed_dict=input_feed)
      #~ self.train_writer.add_summary(summary_str, self.global_step.eval())
    #
    #
    if training_mode:
      # Gradient norm, loss, [outputs_forward, outputs_backward]
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
        return outputs[1], outputs[2], outputs[3:]
      else:
        return outputs[1], outputs[2], outputs[3:(3+decoder_size)], outputs[(3+decoder_size):]
    else:
      # No gradient norm, loss, [outputs_forward, outputs_backward]
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
        return None, outputs[0], outputs[1:]
      else:
        return None, outputs[0], outputs[1:(1+decoder_size)], outputs[(1+decoder_size):]






class DecodingModel(object):


  def __init__(self):
    
    # LSTM cell
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=data_utils.num_units, state_is_tuple=True)
    #~ single_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=data_utils.num_units)
    #~ single_cell = tf.nn.rnn_cell.GRUCell(num_units=data_utils.num_units)
    if (data_utils.num_layers > 1):
	    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * data_utils.num_layers)
    else:
      cell = single_cell
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=1.0,output_keep_prob=1.0)
    
    # INPUT placeholder
    #
    self.input_spectrum = tf.placeholder(tf.float32,shape=[None,data_utils.MZ_SIZE],name="input_spectrum")
    #
    # nobi
    self.input_AA_id = [tf.placeholder(tf.int32,shape=[None],name="input_AA_id_1"),
                        tf.placeholder(tf.int32,shape=[None],name="input_AA_id_2")]
    self.input_intensity = tf.placeholder(tf.float32,
                                          shape=[None,data_utils.vocab_size,data_utils.num_ion,data_utils.WINDOW_SIZE],
                                          name="input_intensity")
    #
    #~ self.input_state = (tf.placeholder(tf.float32,shape=[None,cell.state_size],name="input_state"),
    self.input_state = (tf.placeholder(tf.float32,shape=[None,data_utils.num_units],name="input_c_state"),
                        tf.placeholder(tf.float32,shape=[None,data_utils.num_units],name="input_h_state"))
    
    # OUTPUT
    ((self.lstm_state0_forward,
    self.output_log_prob_forward, 
    self.lstm_state_forward),
    #
    (self.lstm_state0_backward,
    self.output_log_prob_backward,
    self.lstm_state_backward)) = novonet_decoding.decode(self.input_spectrum,
                                                         self.input_AA_id,
                                                         self.input_intensity,
                                                         #~ self.input_state[0],
                                                         self.input_state,
                                                         cell)
        
