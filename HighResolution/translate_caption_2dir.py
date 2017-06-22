from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import gc
import random
import sys
import time
import re

import resource

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import embedding_ops

import data_utils
import novonet_model
from cython_speedup import process_spectrum, get_candidate_intensity











def inspect_file_location(data_format, input_file):

  print("inspect_file_location(), input_file = ", input_file)
  
  if (data_format=="msp"):
    keyword="Name" 
  elif (data_format=="mgf"):
    keyword="BEGIN IONS" 

  spectra_file_location = []
  with open(input_file, mode="r") as file_handle:
    line = True
    while line:
      file_location = file_handle.tell()
      line = file_handle.readline()
      if keyword in line:
        spectra_file_location.append(file_location)
  return spectra_file_location


def read_spectra(file_handle, data_format, spectra_locations):
  
#~   # for testing
#~   test_time = 0.0
  
  print("read_spectra()")
  
  data_set = [[] for _ in data_utils._buckets]

  counter = 0
  counter_skipped = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_skipped_mass = 0
  counter_skipped_mass_precision = 0
  
  avg_peak_count = 0.0
  avg_peptide_len = 0.0

  if (data_format=="mgf"):
    keyword="BEGIN IONS" 

  for location in spectra_locations:
    
    file_handle.seek(location)
    line = file_handle.readline()
    assert (keyword in line), "ERROR: read_spectra(); wrong format"
  
    unknown_modification = False
    max_intensity = 0.0

    # READ AN ENTRY
    if (data_format=="mgf"):

      # header TITLE
      line = file_handle.readline()

      # header PEPMASS
      line = file_handle.readline()
      peptide_ion_mz = float(re.split('=|\n', line)[1])

      # header CHARGE
      line = file_handle.readline()
      charge = float(re.split('=|\+', line)[1])
      
      # header SCANS
      line = file_handle.readline()
#~       scan = int(re.split('=', line)[1])
      scan = re.split('=|\n', line)[1]

      # header RTINSECONDS
      line = file_handle.readline()

      # header SEQ
      line = file_handle.readline()
      raw_sequence = re.split('=|\n|\r', line)[1]
      #~ peptide = peptide.translate(None, '+-.0123456789)') # modification signal "("
      #~ peptide = peptide.translate(None, '(+-.0123456789)') # ignore modifications
      raw_sequence_len = len(raw_sequence)
      peptide = []
      index = 0
      while (index < raw_sequence_len):
        if (raw_sequence[index]=="("):
          if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+57.02)"):
#~           if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+58.01)"): # orbi
            peptide[-1] = "Cmod"
            index += 8
          elif (peptide[-1]=='M' and raw_sequence[index:index+8]=="(+15.99)"):
            peptide[-1] = 'Mmod'
            index += 8
          elif (peptide[-1]=='N' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Nmod'
            index += 6
          elif (peptide[-1]=='Q' and raw_sequence[index:index+6]=="(+.98)"):
            peptide[-1] = 'Qmod'
            index += 6
          else: # unknown modification
          #~ elif ("".join(raw_sequence[index:index+8])=="(+42.01)"):
            #~ print("ERROR: unknown modification!")
            #~ print("raw_sequence = ", raw_sequence)
            #~ sys.exit()
            unknown_modification = True
            break
        else:
          peptide.append(raw_sequence[index])
          index += 1
      #
      if (unknown_modification):
        counter_skipped += 1
        counter_skipped_mod += 1
        continue
        
      # neutral peptide_mass <= MZ_MAX(3000.0) # TEMP
      peptide_mass = peptide_ion_mz*charge - charge*data_utils.mass_H
      if (peptide_mass > data_utils.MZ_MAX):
        counter_skipped += 1
        counter_skipped_mass += 1
        continue

      # TRAINING-ONLY: consider peptide length <= MAX_LEN(20)
      peptide_len = len(peptide)
      if (peptide_len > data_utils.MAX_LEN): 
        #~ print("ERROR: peptide_len {0} exceeds MAX_LEN {1}".format(peptide_len, data_utils.MAX_LEN))
        #~ sys.exit()
        counter_skipped += 1
        counter_skipped_len += 1
        continue

      # TRAINING-ONLY: testing peptide_mass & sequence_mass
      sequence_mass = sum(data_utils.mass_AA[aa] for aa in peptide)
      sequence_mass += data_utils.mass_N_terminus + data_utils.mass_C_terminus
      if (abs(peptide_mass-sequence_mass) > data_utils.PRECURSOR_MASS_PRECISION_INPUT_FILTER):
        #
        #~ print("ERROR: peptide_mass and sequence_mass not matched")
        #~ print("peptide = ", peptide)
        #~ print("peptide_list_mod = ", peptide_list_mod)
        #~ print("peptide_list = ", peptide_list)
        #~ print("peptide_ion_mz = ",peptide_ion_mz)
        #~ print("charge = ", charge)
        #~ print("peptide_mass  ", peptide_mass)
        #~ print("sequence_mass ", sequence_mass)
        #~ sys.exit()
        #
        counter_skipped_mass_precision += 1
        counter_skipped += 1
        continue

      # read spectrum
      spectrum_mz = []
      spectrum_intensity = []
      line = file_handle.readline()
      while not ("END IONS" in line):
        #
        mz, intensity = re.split(' |\n', line)[:2]
        intensity_float = float(intensity)
        mz_float = float(mz)
        #
        if (mz_float > data_utils.MZ_MAX): # skip this peak
          line = file_handle.readline()
          continue
        #
        spectrum_mz.append(mz_float)
        spectrum_intensity.append(intensity_float)
        #
        line = file_handle.readline()

    # AN ENTRY FOUND
    counter += 1
    if counter % 10000 == 0:
      print("  reading peptide %d" % counter)
    
    # Average number of peaks per spectrum
    peak_count = len(spectrum_mz)
    avg_peak_count += peak_count

    # Average peptide length
    avg_peptide_len += peptide_len
    
    
    
    
    
    # PRE-PROCESS SPECTRUM
    spectrum_holder, spectrum_original_forward, spectrum_original_backward = process_spectrum(
                                                                               spectrum_mz,
                                                                               spectrum_intensity,
                                                                               peptide_mass)






    # PRE-PROCESS decoder_input
    #
    for bucket_id, target_size in enumerate(data_utils._buckets):
      if peptide_len + 2 <= target_size: # +2 to include GO and EOS
        break
    decoder_size = data_utils._buckets[bucket_id]
    #
    # parse peptide AA sequence to list of ids
    peptide_ids = [data_utils.vocab[x] for x in peptide]
    #
    # PADDING
    pad_size = decoder_size - (len(peptide_ids) + 2)

    # forward
    if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==2):
      peptide_ids_forward = peptide_ids[:]
      peptide_ids_forward.insert(0, data_utils.GO_ID)
      peptide_ids_forward.append(data_utils.EOS_ID)
      peptide_ids_forward += [data_utils.PAD_ID] * pad_size

    # backward
    if (data_utils.FLAGS.direction==1 or data_utils.FLAGS.direction==2):
      peptide_ids_backward = peptide_ids[::-1]
      peptide_ids_backward.insert(0, data_utils.EOS_ID)
      peptide_ids_backward.append(data_utils.GO_ID)
      peptide_ids_backward += [data_utils.PAD_ID] * pad_size







#~     # for testing
#~     start_time = time.time()
  
    # PRE-PROCESS candidate_intensity
    if (not data_utils.FLAGS.beam_search):

      # forward
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==2):
  
        candidate_intensity_list_forward = []
        prefix_mass = 0.0
  
        for index in xrange(decoder_size):
          #
          prefix_mass += data_utils.mass_ID[peptide_ids_forward[index]]
          #
          candidate_intensity = get_candidate_intensity(spectrum_original_forward,
                                                        peptide_mass,
                                                        prefix_mass,
                                                        direction=0)
          #
          candidate_intensity_list_forward.append(candidate_intensity)
      
      # backward
      if (data_utils.FLAGS.direction==1 or data_utils.FLAGS.direction==2):
  
        candidate_intensity_list_backward = []
        suffix_mass = 0.0
  
        for index in xrange(decoder_size):
          #
          suffix_mass += data_utils.mass_ID[peptide_ids_backward[index]]
          #
          candidate_intensity = get_candidate_intensity(spectrum_original_backward,
                                                        peptide_mass,
                                                        suffix_mass,
                                                        direction=1)
          #
          candidate_intensity_list_backward.append(candidate_intensity)
      
#~     # for testing
#~     test_time += time.time() - start_time
      
    
    
    
    

    # assign data to buckets
    if (data_utils.FLAGS.beam_search):
      if (data_utils.FLAGS.direction==0):
        data_set[bucket_id].append([scan, spectrum_holder, spectrum_original_forward, peptide_mass, peptide_ids_forward])
      elif (data_utils.FLAGS.direction==1):
        data_set[bucket_id].append([scan, spectrum_holder, spectrum_original_backward, peptide_mass, peptide_ids_backward])
      else:
        data_set[bucket_id].append([scan,
                                    spectrum_holder, 
                                    spectrum_original_forward, 
                                    spectrum_original_backward, 
                                    peptide_mass, 
                                    peptide_ids_forward,
                                    peptide_ids_backward])
    else:
      if (data_utils.FLAGS.direction==0):
        data_set[bucket_id].append([spectrum_holder, candidate_intensity_list_forward, peptide_ids_forward])
      elif (data_utils.FLAGS.direction==1):
        data_set[bucket_id].append([spectrum_holder, candidate_intensity_list_backward, peptide_ids_backward])
      else:
        data_set[bucket_id].append([spectrum_holder, 
                                    candidate_intensity_list_forward, 
                                    candidate_intensity_list_backward, 
                                    peptide_ids_forward,
                                    peptide_ids_backward])

#~   # for testing
#~   print("test_time = {0:.2f}".format(test_time))

  print("  total peptide read %d" % counter)
  print("  total peptide skipped %d" % counter_skipped)
  print("  total peptide skipped by mod %d" % counter_skipped_mod)
  print("  total peptide skipped by len %d" % counter_skipped_len)
  print("  total peptide skipped by mass %d" % counter_skipped_mass)
  print("  total peptide skipped by mass precision %d" % counter_skipped_mass_precision)

  print("  average #peaks per spectrum %.1f" % (avg_peak_count/counter))
  print("  average peptide length %.1f" % (avg_peptide_len/counter))
  
  return data_set, counter


def read_random_stack(file_handle, data_format, spectra_locations, stack_size):

  print("read_random_stack()")
  
  random_locations = random.sample(spectra_locations, min(stack_size, len(spectra_locations)))
  
  return read_spectra(file_handle, data_format, random_locations)


def get_batch_01(index_list, data_set, bucket_id):
    #~ print("get_batch()")
    
    batch_size = len(index_list)
    
    # SPECTRA
    spectra_list = []

    # candidate_intensity
    candidate_intensity_lists = []
    
    # PEPTIDE
    decoder_inputs = []

    for batch, index in enumerate(index_list):
      
      # Get a random entry of encoder and decoder inputs from data,
      spectrum_holder, candidate_intensity_list, decoder_input = data_set[bucket_id][index]
      #
      spectra_list.append(spectrum_holder)
      #
      candidate_intensity_lists.append(candidate_intensity_list)
      #
      decoder_inputs.append(decoder_input)
        
    batch_encoder_inputs = [np.array(spectra_list)]
    batch_intensity_inputs = []
    batch_decoder_inputs = []
    batch_weights = []
    #
    # batch_decoder_inputs are just re-indexed decoder_inputs.
    decoder_size = data_utils._buckets[bucket_id]
    for length_idx in xrange(decoder_size):
      #
      batch_intensity_inputs.append(
          np.array([candidate_intensity_lists[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.float32))
      #
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if (length_idx == decoder_size - 1 or 
            target == data_utils.EOS_ID or
            target == data_utils.GO_ID or
            target == data_utils.PAD_ID):
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_intensity_inputs, batch_decoder_inputs, batch_weights


def get_batch_2(index_list, data_set, bucket_id):
    #~ print("get_batch()")
    
    batch_size = len(index_list)
    
    # SPECTRA
    spectra_list = []

    # candidate_intensity
    candidate_intensity_lists_forward = []
    candidate_intensity_lists_backward = []
    
    # PEPTIDE
    decoder_inputs_forward = []
    decoder_inputs_backward = []

    for batch, index in enumerate(index_list):
      
      # Get a random entry of encoder and decoder inputs from data,
      spectrum_holder, candidate_intensity_list_forward, candidate_intensity_list_backward, decoder_input_forward, decoder_input_backward = data_set[bucket_id][index]
      #
      spectra_list.append(spectrum_holder)
      #
      candidate_intensity_lists_forward.append(candidate_intensity_list_forward)
      candidate_intensity_lists_backward.append(candidate_intensity_list_backward)
      #
      decoder_inputs_forward.append(decoder_input_forward)
      decoder_inputs_backward.append(decoder_input_backward)
        
    batch_encoder_inputs = [np.array(spectra_list)]
    batch_intensity_inputs_forward = []
    batch_intensity_inputs_backward = []
    batch_decoder_inputs_forward = []
    batch_decoder_inputs_backward = []
    batch_weights = []
    #
    # batch_decoder_inputs are just re-indexed decoder_inputs.
    decoder_size = data_utils._buckets[bucket_id]
    for length_idx in xrange(decoder_size):
      #
      batch_intensity_inputs_forward.append(
          np.array([candidate_intensity_lists_forward[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.float32))
      #
      batch_intensity_inputs_backward.append(
          np.array([candidate_intensity_lists_backward[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.float32))
      #
      batch_decoder_inputs_forward.append(
          np.array([decoder_inputs_forward[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))
      #
      batch_decoder_inputs_backward.append(
          np.array([decoder_inputs_backward[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs_forward[batch_idx][length_idx + 1]
        if (length_idx == decoder_size - 1 or 
            target == data_utils.EOS_ID or
            target == data_utils.GO_ID or
            target == data_utils.PAD_ID):
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_intensity_inputs_forward, batch_intensity_inputs_backward, batch_decoder_inputs_forward, batch_decoder_inputs_backward, batch_weights


def create_model(session, training_mode):
  """Create translation model and initialize or load parameters in session."""

  print("create_model()")

  model = novonet_model.TrainingModel(session, training_mode)

  # folder of training state
  ckpt = tf.train.get_checkpoint_state(data_utils.FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
#~     train_writer = tf.train.SummaryWriter("train_log", session.graph)
#~     train_writer.close()
    session.run(tf.initialize_all_variables())
  return model


def trim_decoder_input(decoder_input, direction):

  if (direction==0):
    LAST_LABEL = data_utils.EOS_ID
  elif (direction==1):
    LAST_LABEL = data_utils.GO_ID
  
  # excluding FIRST_LABEL, LAST_LABEL & PAD
  return decoder_input[1:decoder_input.index(LAST_LABEL)]


def print_AA_basic(output_file_handle, scan, decoder_input, output, direction,
                   accuracy_AA, len_AA, exact_match):
  
  if (direction==0):
    LAST_LABEL = data_utils.EOS_ID
  elif (direction==1):
    LAST_LABEL = data_utils.GO_ID
  
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_AA = [data_utils.vocab_reverse[x] for x in decoder_input]
  
  output_seq, output_score = output
  output_len = output_seq.index(LAST_LABEL) if LAST_LABEL in output_seq else 0
  output_AA = [data_utils.vocab_reverse[x] for x in output_seq[:output_len]]
  
  if (direction==1):
    decoder_input_AA = decoder_input_AA[::-1]
    output_AA = output_AA[::-1]

  print ("%s\t%s\t%s\t%.2f\t%d\t%d\t%d\n"
         % (scan, ",".join(decoder_input_AA), ",".join(output_AA), output_score,
            accuracy_AA, len_AA, exact_match), 
         file=output_file_handle,
         end="")


def test_AA_match_1by1(decoder_input, output):

  decoder_input_len = len(decoder_input)

  num_match = 0
  
  index_aa = 0
  while index_aa < decoder_input_len:
    #~ if  decoder_input[index_aa]==output[index_aa]:
    if abs(data_utils.mass_ID[decoder_input[index_aa]]-data_utils.mass_ID[output[index_aa]]) < data_utils.AA_MATCH_PRECISION:
      num_match += 1
    index_aa += 1
  
  return num_match


def test_AA_match_novor(decoder_input, output):

  num_match = 0
  
  decoder_input_len = len(decoder_input)
  output_len = len(output)

  decoder_input_mass = [data_utils.mass_ID[x] for x in decoder_input]
  decoder_input_mass_cum = np.cumsum(decoder_input_mass)

  output_mass = [data_utils.mass_ID[x] for x in output]
  output_mass_cum = np.cumsum(output_mass)
  
  i = 0
  j = 0
  while (i < decoder_input_len and j < output_len):

    #~ # for testing
    #~ print(decoder_input_mass_cum[i])
    #~ print(output_mass_cum[j])

    if (abs(decoder_input_mass_cum[i] - output_mass_cum[j]) < 0.5):

      #~ # for testing
      #~ print(decoder_input_mass[i] )
      #~ print(output_mass[j])

      if (abs(decoder_input_mass[i] - output_mass[j]) < 0.1):
      #~ if  decoder_input[index_aa]==output[index_aa]:
        num_match += 1

      i += 1
      j += 1
    elif (decoder_input_mass_cum[i] < output_mass_cum[j]):
      i += 1
    else:
      j += 1

    #~ # for testing
    #~ print(num_match)
    
  return num_match


def test_AA_decode_single(decoder_input, output, direction):
  
  accuracy_AA = 0.0
  len_AA = 0.0
  exact_match = 0.0
  len_match = 0.0
  
  if (direction==0):
    LAST_LABEL = data_utils.EOS_ID
  elif (direction==1):
    LAST_LABEL = data_utils.GO_ID
  
  # decoder_input = [AA]; output = [AA]
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_len = len(decoder_input)
  #
  output_len = output.index(LAST_LABEL) if LAST_LABEL in output else 0
  output = output[:output_len]
  
  # measure accuracy
  num_match = test_AA_match_novor(decoder_input, output)

  #~ accuracy_AA = num_match / decoder_input_len
  accuracy_AA = num_match
  len_AA = decoder_input_len
  len_decode = output_len

  if (num_match==decoder_input_len):
    exact_match = 1.0

  if (output_len==decoder_input_len):
    len_match = 1.0

  # testing
  #~ print(decoder_input_AA)
  #~ print(output_AA)
  #~ print(num_match)
  #~ print(batch_accuracy_AA)
  #~ print(num_exact_match)
  #~ print(num_len_match)
  #~ sys.exit()
  
  return accuracy_AA, len_AA, len_decode, exact_match, len_match


def test_AA_true_feeding_single(decoder_input, output, direction):
  
  accuracy_AA = 0.0
  len_AA = 0.0
  exact_match = 0.0
  len_match = 0.0
  
  if (direction==0):
    LAST_LABEL = data_utils.EOS_ID
  elif (direction==1):
    LAST_LABEL = data_utils.GO_ID
  
  # decoder_input = [AA]; output = [AA...]
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_len = len(decoder_input)
  
  # measure accuracy
  num_match = test_AA_match_1by1(decoder_input, output)

  #~ accuracy_AA = num_match / decoder_input_len
  accuracy_AA = num_match
  len_AA = decoder_input_len

  if (num_match==decoder_input_len):
    exact_match = 1.0

  #~ if (output_len==decoder_input_len):
    #~ len_match = 1.0

  return accuracy_AA, len_AA, exact_match, len_match


def test_AA_decode_batch(scans, decoder_inputs, outputs, direction, output_file_handle):

  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  batch_len_decode = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0
  
  batch_size = len(decoder_inputs)
  
  for index in xrange(len(scans)):
  #~ # for testing
  #~ for index in xrange(15,20):

    scan = scans[index]
    decoder_input = decoder_inputs[index] 
    output = outputs[index]

    output_seq, output_score = output

    accuracy_AA, len_AA, len_decode, exact_match, len_match = test_AA_decode_single(decoder_input, output_seq, direction)
    
    
    #~ # for testing
    #~ print([data_utils.vocab_reverse[x] for x in decoder_input[1:]])
    #~ print([data_utils.vocab_reverse[x] for x in output_seq])
    #~ print(accuracy_AA)
    #~ print(exact_match)
    #~ print(len_match)

  # print to output file
    print_AA_basic(output_file_handle, scan, decoder_input, output, direction,
                   accuracy_AA, len_AA, exact_match)
    
    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    batch_len_decode += len_decode
    num_exact_match += exact_match
    num_len_match += len_match

  #~ return batch_accuracy_AA/batch_size, num_exact_match/batch_size, num_len_match/batch_size
#~   return batch_accuracy_AA/batch_len_AA, num_exact_match/batch_size, num_len_match/batch_size
  return batch_accuracy_AA, batch_len_AA, batch_len_decode, num_exact_match, num_len_match


def test_logit_single_01(decoder_input, output_logit):
  
  output = [np.argmax(x) for x in output_logit]

  return test_AA_true_feeding_single(decoder_input, output, data_utils.FLAGS.direction)


def test_logit_single_2(decoder_input_forward,
                        decoder_input_backward,
                        output_logit_forward,
                        output_logit_backward):
  
  # length excluding FIRST_LABEL & LAST_LABEL
  decoder_input_len = decoder_input_forward[1:].index(data_utils.EOS_ID)
  
  # average forward-backward prediction logit
  logit_forward = output_logit_forward[:decoder_input_len]
  logit_backward = output_logit_backward[:decoder_input_len]
  logit_backward = logit_backward[::-1]
  output = []
  for x,y in zip(logit_forward, logit_backward):
    prob_forward = np.exp(x)/np.sum(np.exp(x))
    prob_backward = np.exp(y)/np.sum(np.exp(y))
    output.append(np.argmax(prob_forward * prob_backward))
    
  output.append(data_utils.EOS_ID)

  return test_AA_true_feeding_single(decoder_input_forward, output, direction=0)


def test_logit_batch_01(decoder_inputs, output_logits):

  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0
  
  batch_size = len(decoder_inputs[0])
  
  for batch in xrange(batch_size):

    decoder_input = [x[batch] for x in decoder_inputs] 
    
    output_logit = [x[batch] for x in output_logits]
    
    accuracy_AA, len_AA, exact_match, len_match = test_logit_single_01(decoder_input, output_logit)
    
    # for testing
    #~ if (exact_match==0):
      #~ print(batch)
      #~ print([data_utils.vocab_reverse[x] for x in decoder_input[1:]])
      #~ print([data_utils.vocab_reverse[np.argmax(x)] for x in output_logit])
      #~ print(accuracy_AA)
      #~ print(exact_match)
      #~ print(len_match)
      #~ sys.exit()
    
    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    num_exact_match += exact_match
    num_len_match += len_match
      
  #~ return batch_accuracy_AA/batch_size, num_exact_match/batch_size, num_len_match/batch_size
  #~ return batch_accuracy_AA/batch_len_AA, num_exact_match/batch_size, num_len_match/batch_size
  return batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match


def test_logit_batch_2(decoder_inputs_forward, 
                          decoder_inputs_backward, 
                          output_logits_forward,
                          output_logits_backward):
                            
  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0
  
  batch_size = len(decoder_inputs_forward[0])
  
  for batch in xrange(batch_size):

    decoder_input_forward = [x[batch] for x in decoder_inputs_forward] 
    decoder_input_backward = [x[batch] for x in decoder_inputs_backward] 
    
    output_logit_forward = [x[batch] for x in output_logits_forward]
    output_logit_backward = [x[batch] for x in output_logits_backward]
    
    accuracy_AA, len_AA, exact_match, len_match = test_logit_single_2(decoder_input_forward,
                                                                      decoder_input_backward, 
                                                                      output_logit_forward,
                                                                      output_logit_backward)
    
    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    num_exact_match += exact_match
    num_len_match += len_match
      
  #~ return batch_accuracy_AA/batch_size, num_exact_match/batch_size, num_len_match/batch_size
  #~ return batch_accuracy_AA/batch_len_AA, num_exact_match/batch_size, num_len_match/batch_size
  return batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match


def test_random_accuracy(sess, model, data_set, bucket_id):
    #~ print("test_random_accuracy()")

    step_time = 0.0
    step_loss = 0.0
    step_accuracy_AA = 0.0
    #~ step_len_AA = 0.0
    step_accuracy_peptide = 0.0
    step_accuracy_len = 0.0
    
    data_set_len = len(data_set[bucket_id])
    
    num_step = data_utils.random_test_batches

    for step in xrange(num_step):

        start_time = time.time()

        random_index_list = random.sample(xrange(data_set_len), data_utils.batch_size)

        # get_batch_01/2
        if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
          encoder_inputs, intensity_inputs, decoder_inputs, target_weights = get_batch_01(random_index_list, 
                                                                                          data_set, 
                                                                                          bucket_id)
        else:
          encoder_inputs, intensity_inputs_forward, intensity_inputs_backward, decoder_inputs_forward, decoder_inputs_backward, target_weights = get_batch_2(random_index_list, data_set, bucket_id)

        # model_step
        if (data_utils.FLAGS.direction==0):
          _, step_loss, output_logits = model.step(sess, encoder_inputs, 
                                                   intensity_inputs_forward=intensity_inputs,
                                                   decoder_inputs_forward=decoder_inputs, 
                                                   target_weights=target_weights,
                                                   bucket_id=bucket_id, 
                                                   training_mode=False)
        elif (data_utils.FLAGS.direction==1):
          _, step_loss, output_logits = model.step(sess, encoder_inputs, 
                                                   intensity_inputs_backward=intensity_inputs,
                                                   decoder_inputs_backward=decoder_inputs, 
                                                   target_weights=target_weights,
                                                   bucket_id=bucket_id, 
                                                   training_mode=False)
        else:
          _, step_loss, output_logits_forward, output_logits_backward = model.step(sess, encoder_inputs, 
                                                                                   intensity_inputs_forward=intensity_inputs_forward,
                                                                                   intensity_inputs_backward=intensity_inputs_backward,
                                                                                   decoder_inputs_forward=decoder_inputs_forward, 
                                                                                   decoder_inputs_backward=decoder_inputs_backward, 
                                                                                   target_weights=target_weights,
                                                                                   bucket_id=bucket_id, 
                                                                                   training_mode=False)
        
        step_time += (time.time() - start_time)
        step_loss += step_loss
        
        # test_logit_batch_01/2
        if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
          batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_01(
                                                                              decoder_inputs, 
                                                                              output_logits)
        else:
          batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_2(
                                                                              decoder_inputs_forward,
                                                                              decoder_inputs_backward, 
                                                                              output_logits_forward,
                                                                              output_logits_backward)

        step_accuracy_AA += batch_accuracy_AA/batch_len_AA
        step_accuracy_peptide += num_exact_match/data_utils.batch_size
        step_accuracy_len += num_len_match/data_utils.batch_size

    # Note that we calculate the average accuracy over STEPS
    #     because it doesn't make sense to pool batches of different steps
    step_time /= num_step
    step_loss /= num_step
    step_accuracy_AA /= num_step
    step_accuracy_peptide /= num_step
    step_accuracy_len /= num_step
    eval_ppx = math.exp(step_loss) if step_loss < 300 else float('inf')
    
    return eval_ppx, step_accuracy_AA, step_accuracy_peptide, step_accuracy_len


def test_accuracy(sess, model, data_set, bucket_id, print_summary=True):

    spectrum_time = 0.0
    avg_loss = 0.0
    avg_accuracy_AA = 0.0
    avg_len_AA = 0.0
    avg_accuracy_peptide = 0.0
    avg_accuracy_len = 0.0
    
    data_set_len = len(data_set[bucket_id])
    data_set_index_list = range(data_set_len)
    data_set_index_chunk_list = [data_set_index_list[i:i+data_utils.batch_size] 
                                 for i in range(0, data_set_len, data_utils.batch_size)]

    for chunk in data_set_index_chunk_list:

      start_time = time.time()

      # get_batch_01/2
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
        encoder_inputs, intensity_inputs, decoder_inputs, target_weights = get_batch_01(chunk, 
                                                                                        data_set, 
                                                                                        bucket_id)
      else:
        encoder_inputs, intensity_inputs_forward, intensity_inputs_backward, decoder_inputs_forward, decoder_inputs_backward, target_weights = get_batch_2(chunk, data_set, bucket_id)
  
      # model_step
      if (data_utils.FLAGS.direction==0):
        _, loss, output_logits = model.step(sess, encoder_inputs, 
                                            intensity_inputs_forward=intensity_inputs,
                                            decoder_inputs_forward=decoder_inputs, 
                                            target_weights=target_weights,
                                            bucket_id=bucket_id, 
                                            training_mode=False)
      elif (data_utils.FLAGS.direction==1):
        _, loss, output_logits = model.step(sess, encoder_inputs, 
                                            intensity_inputs_backward=intensity_inputs,
                                            decoder_inputs_backward=decoder_inputs, 
                                            target_weights=target_weights,
                                            bucket_id=bucket_id, 
                                            training_mode=False)
      else:
        _, loss, output_logits_forward, output_logits_backward = model.step(sess, encoder_inputs, 
                                                                            intensity_inputs_forward=intensity_inputs_forward,
                                                                            intensity_inputs_backward=intensity_inputs_backward,
                                                                            decoder_inputs_forward=decoder_inputs_forward, 
                                                                            decoder_inputs_backward=decoder_inputs_backward, 
                                                                            target_weights=target_weights,
                                                                            bucket_id=bucket_id, 
                                                                            training_mode=False)
          
      spectrum_time += time.time() - start_time
  
      # test_logit_batch_01/2
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
        batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_01(
                                                                            decoder_inputs, 
                                                                            output_logits)
      else:
        batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_2(
                                                                            decoder_inputs_forward,
                                                                            decoder_inputs_backward, 
                                                                            output_logits_forward,
                                                                            output_logits_backward)

      avg_loss += loss * len(chunk) # because the loss has been averaged over batch_size
      avg_accuracy_AA += batch_accuracy_AA
      avg_len_AA += batch_len_AA
      avg_accuracy_peptide += num_exact_match
      avg_accuracy_len += num_len_match
      
    spectrum_time /= data_set_len
    avg_loss /= data_set_len
    avg_accuracy_AA /= avg_len_AA
    avg_accuracy_peptide /= data_set_len
    avg_accuracy_len /= data_set_len
  
    eval_ppx = math.exp(avg_loss) if avg_loss < 300 else float('inf')
    
    if (print_summary):
      print("test_accuracy()")
      print("  bucket %d spectrum_time %.4f" % (bucket_id, spectrum_time))
      print("  bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
      print("  bucket %d avg_accuracy_AA %.4f" % (bucket_id, avg_accuracy_AA))
      print("  bucket %d avg_accuracy_peptide %.4f" % (bucket_id, avg_accuracy_peptide))
      print("  bucket %d avg_accuracy_len %.4f" % (bucket_id, avg_accuracy_len))

    return avg_loss, avg_accuracy_AA, avg_accuracy_peptide, avg_accuracy_len


def knapsack_example():
  
  peptide_mass = 11
  print("peptide_mass = ", peptide_mass)
  mass_aa = [2,3,4,5]
  print("mass_aa = ", mass_aa)
  
  knapsack_matrix = np.zeros(shape=(4,11),dtype=bool)
  
  for aa_id in xrange(4):
    #
    for col in xrange(peptide_mass):
      #
      current_mass = col + 1
      #
      if (current_mass < mass_aa[aa_id]):
        knapsack_matrix[aa_id,col] = False
      #
      if (current_mass == mass_aa[aa_id]):
        knapsack_matrix[aa_id,col] = True
      #
      if (current_mass > mass_aa[aa_id]):
        #
        sub_mass = current_mass - mass_aa[aa_id]
        sub_col = sub_mass - 1
        #
        if (np.sum(knapsack_matrix[:,sub_col]) > 0):
          knapsack_matrix[aa_id,col] = True
          knapsack_matrix[:,col] = np.logical_or(knapsack_matrix[:,col],
                                                 knapsack_matrix[:,sub_col])
        else:
          knapsack_matrix[aa_id,col] = False

    print("mass_aa[{0}] = {1}".format(aa_id,mass_aa[aa_id]))
    print(knapsack_matrix)
      

def knapsack_build():
  
  peptide_mass = data_utils.MZ_MAX
  peptide_mass = peptide_mass - (data_utils.mass_C_terminus + data_utils.mass_H)
  print("peptide_mass = ", peptide_mass)
  #
  peptide_mass_round = int(round(peptide_mass * data_utils.KNAPSACK_AA_RESOLUTION))
  print("peptide_mass_round = ", peptide_mass_round)
  #
  #~ peptide_mass_upperbound = peptide_mass_round + data_utils.KNAPSACK_MASS_PRECISION_TOLERANCE 
  peptide_mass_upperbound = peptide_mass_round + data_utils.KNAPSACK_AA_RESOLUTION 

  knapsack_matrix = np.zeros(shape=(data_utils.vocab_size,peptide_mass_upperbound),dtype=bool)
  
  for aa_id in xrange(3,data_utils.vocab_size): # excluding PAD, GO, EOS
    #
    mass_aa_round = int(round(data_utils.mass_ID[aa_id] * data_utils.KNAPSACK_AA_RESOLUTION))
    print(data_utils.vocab_reverse[aa_id], mass_aa_round)
    #
    for col in xrange(peptide_mass_upperbound):
      #
      # col 0 ~ mass 1
      # col + 1 = mass
      # col = mass - 1
      current_mass = col + 1
      #
      if (current_mass < mass_aa_round):
        knapsack_matrix[aa_id,col] = False
      #
      if (current_mass == mass_aa_round):
        knapsack_matrix[aa_id,col] = True
      #
      if (current_mass > mass_aa_round):
        #
        sub_mass = current_mass - mass_aa_round
        sub_col = sub_mass - 1
        #
        if (np.sum(knapsack_matrix[:,sub_col]) > 0):
          knapsack_matrix[aa_id,col] = True
          knapsack_matrix[:,col] = np.logical_or(knapsack_matrix[:,col],
                                                 knapsack_matrix[:,sub_col])
        else:
          knapsack_matrix[aa_id,col] = False

  np.save("knapsack.npy",knapsack_matrix)


def knapsack_search(knapsack_matrix, peptide_mass, mass_precision_tolerance):

  #~ knapsack_matrix = np.load("knapsack.npy")
  
  peptide_mass_round = int(round(peptide_mass * data_utils.KNAPSACK_AA_RESOLUTION))
  #
  peptide_mass_upperbound = peptide_mass_round + mass_precision_tolerance # 100 x 0.0001 Da
  peptide_mass_lowerbound = peptide_mass_round - mass_precision_tolerance
  #
  if (peptide_mass_upperbound < data_utils.mass_AA_min_round): # 57.0215 
    return []
  #
  # [peptide_mass_lowerbound, peptide_mass_upperbound] will NOT be less than mass_AA_min_round.
  # peptide_mass_upperbound may exceed column 2982.9895, but numpy will ignore the extra indices.
  #
  #~ if (peptide_mass_lowerbound < 0):
    #~ return []
  #
  # col 0 ~ mass 1
  # col + 1 = mass
  # col = mass - 1
  # [)
  peptide_mass_lowerbound_col = peptide_mass_lowerbound - 1
  peptide_mass_upperbound_col = peptide_mass_upperbound - 1

  # Search for any nonzero col
  candidate_AA_id = np.flatnonzero(np.any(knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1],
                                          axis=1))

  # Search for closest col
  #~ m = knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1]
  #~ m_any = np.any(m, axis=0)
  #~ m_nonzero_col = np.flatnonzero(m_any)
  #~ if (m_nonzero_col.size == 0):
    #~ return []
  #~ m_closest_col = m_nonzero_col[np.argmin(np.absolute(m_nonzero_col-mass_precision_tolerance))]
  #~ candidate_AA_id = np.flatnonzero(m[:, m_closest_col])

  return candidate_AA_id.tolist()
  
  
  
  


  # EXAMPLE
  aa_list = ['_GO','S','E','V','V','I','Cmod','S','H','D','K']
  print("EXAMPLE")
  print("aa_list = ", aa_list)
  #
  peptide_mass = 391.85632 * 3 - 3 * 1.0078
  peptide_mass = peptide_mass - (data_utils.mass_C_terminus + data_utils.mass_H)
  print("peptide_mass = ", peptide_mass)
  #
  suffix_mass = peptide_mass
  #
  for aa in aa_list[:-1]:
    #
    suffix_mass -= data_utils.mass_AA[aa]
    print("suffix_mass = ", suffix_mass)
    #
    suffix_mass_col = int(round(suffix_mass * data_utils.KNAPSACK_AA_RESOLUTION)) - 1
    print("suffix_mass_col = ", suffix_mass_col)
    #
    suffix_mass_upperbound = suffix_mass_col + mass_precision_tolerance + 1 
    suffix_mass_lowerbound = suffix_mass_col - mass_precision_tolerance
    #
    candidate_AA_id = np.flatnonzero(np.any(knapsack_matrix[:,suffix_mass_lowerbound:suffix_mass_upperbound],
                                            axis=1))
    candidate_AA = [data_utils.vocab_reverse[x] for x in candidate_AA_id]
    print(candidate_AA)
    print(len(candidate_AA))
      

def knapsack_search_mass(knapsack_matrix, peptide_mass, mass_precision_tolerance):
  
  #~ knapsack_matrix = np.load("knapsack.npy")
  
  peptide_mass_round = int(round(peptide_mass * data_utils.KNAPSACK_AA_RESOLUTION))
  #
  peptide_mass_upperbound = peptide_mass_round + mass_precision_tolerance # 100 x 0.0001 Da
  peptide_mass_lowerbound = peptide_mass_round - mass_precision_tolerance
  #
  if (peptide_mass_upperbound < data_utils.mass_AA_min_round): # 57.0215 
    return []
  #
  # [peptide_mass_lowerbound, peptide_mass_upperbound] will NOT be less than mass_AA_min_round.
  # peptide_mass_upperbound may exceed column 2982.9895, but numpy will ignore the extra indices.
  #
  #~ if (peptide_mass_lowerbound < 0):
    #~ return []
  #
  # col 0 ~ mass 1
  # col + 1 = mass
  # col = mass - 1
  # [)
  peptide_mass_lowerbound_col = peptide_mass_lowerbound - 1
  peptide_mass_upperbound_col = peptide_mass_upperbound - 1

  # Search for any nonzero col
  #~ candidate_AA_id = np.flatnonzero(np.any(knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1],
                                          #~ axis=1))

  # Search for closest col
  m = knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1]
  m_any = np.any(m, axis=0)
  m_nonzero_col = np.flatnonzero(m_any)
  if (m_nonzero_col.size == 0):
    return []
  m_closest_col = m_nonzero_col[np.argmin(np.absolute(m_nonzero_col-mass_precision_tolerance))]
  m_closest_col_mass = (peptide_mass_lowerbound_col + m_closest_col + 1) / float(data_utils.KNAPSACK_AA_RESOLUTION)
  #~ print(m_closest_col_mass)
  #~ sys.exit()

  return [m_closest_col_mass]
      

def decode_true_feeding_01(sess, model, direction, data_set):

  # FORWARD/BACKWARD setting
  if (direction==0):
    model_lstm_state0 = model.lstm_state0_forward
    model_output_log_prob = model.output_log_prob_forward
    model_lstm_state = model.lstm_state_forward
  elif (direction==1):
    model_lstm_state0 = model.lstm_state0_backward
    model_output_log_prob = model.output_log_prob_backward
    model_lstm_state = model.lstm_state_backward

  data_set_len = len(data_set)
  # recall that a data_set[spectrum_id] includes the following
  #     spectrum_holder                     # 0
  #     candidate_intensity_list[MAX_LEN]   # 1
  #     decoder_input[MAX_LEN]              # 2
  
  # process in stacks
  decode_stack_size = 128
  data_set_index_list = range(data_set_len)
  data_set_index_stack_list = [data_set_index_list[i:i+decode_stack_size] 
                               for i in range(0, data_set_len, decode_stack_size)]






  # spectrum_holder >> lstm_state0;
  block_c_state0 = []
  block_h_state0 = []
  #
  for stack in data_set_index_stack_list:
    block_spectrum = np.array([data_set[x][0] for x in stack])
    input_feed = {}
    input_feed[model.input_spectrum.name] = block_spectrum
    output_feed = model_lstm_state0
    stack_c_state0, stack_h_state0 = sess.run(fetches=output_feed, feed_dict=input_feed)
    block_c_state0.append(stack_c_state0)
    block_h_state0.append(stack_h_state0)
  #
  #~ block_state0 = np.vstack(block_state0)






  # MAIN decoding LOOP in STACKS
  output_log_probs = [[] for x in xrange(len(data_set[0][2]))]
  
  for stack_index, stack in enumerate(data_set_index_stack_list):
    
    stack_c_state = block_c_state0[stack_index]
    stack_h_state = block_h_state0[stack_index]

    for index in xrange(len(data_set[0][2])):

      block_candidate_intensity = np.array([data_set[x][1][index] for x in stack])
      block_AA_id_2 = np.array([data_set[x][2][index] for x in stack]) # nobi
      if (index-1 >= 0):
        block_AA_id_1 = np.array([data_set[x][2][index-1] for x in stack])
      else:
        block_AA_id_1 = block_AA_id_2 # nobi
  
      # FEED & RUN
      input_feed = {}
      input_feed[model.input_AA_id[0].name] = block_AA_id_1 # nobi
      input_feed[model.input_AA_id[1].name] = block_AA_id_2 # nobi
      input_feed[model.input_intensity.name] = block_candidate_intensity
      #~ input_feed[model.input_state[0].name] = stack_c_state
      #~ input_feed[model.input_state[1].name] = stack_h_state
      input_feed[model.input_state[0].name] = block_c_state0[stack_index] # nobi
      input_feed[model.input_state[1].name] = block_h_state0[stack_index] # nobi
      #
      #~ output_feed = [model_output_log_prob, model_lstm_state]
      output_feed = model_output_log_prob # nobi
      #
      #~ stack_log_prob, (stack_c_state, stack_h_state) = sess.run(fetches=output_feed,
                                                                #~ feed_dict=input_feed)
      stack_log_prob = sess.run(fetches=output_feed, feed_dict=input_feed) # nobi
      #
      output_log_probs[index].append(stack_log_prob)
      
  output_log_probs = [np.vstack(x) for x in output_log_probs]
  
  return output_log_probs


def decode_true_feeding_2(sess, model, data_set):
  
  data_set_forward = [[x[0],x[1],x[3]] for x in data_set]
  data_set_backward = [[x[0],x[2],x[4]] for x in data_set]
  
  output_log_prob_forwards = decode_true_feeding_01(sess, 
                                                    model, 
                                                    direction=0,
                                                    data_set=data_set_forward)

  output_log_prob_backwards = decode_true_feeding_01(sess, 
                                                     model, 
                                                     direction=1,
                                                     data_set=data_set_backward)
  
  return output_log_prob_forwards, output_log_prob_backwards


def decode_true_feeding(sess, model, data_set):

  spectrum_time = time.time()

  if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
    output_log_probs = decode_true_feeding_01(sess, 
                                              model, 
                                              data_utils.FLAGS.direction, 
                                              data_set)
    spectrum_time = time.time() - spectrum_time
    #
    decoder_inputs = [x[-1] for x in data_set]
    batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_01(
                                                                        zip(*decoder_inputs), 
                                                                        output_log_probs)
  else:
    output_log_prob_forwards, output_log_prob_backwards = decode_true_feeding_2(sess, 
                                                                                model, 
                                                                                data_set)
    spectrum_time = time.time() - spectrum_time
    #
    decoder_inputs_forward = [x[-2] for x in data_set]
    decoder_inputs_backward = [x[-1] for x in data_set]
    batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match = test_logit_batch_2(
                                                                        zip(*decoder_inputs_forward), 
                                                                        zip(*decoder_inputs_backward),
                                                                        output_log_prob_forwards,
                                                                        output_log_prob_backwards)

  return batch_accuracy_AA/batch_len_AA, num_exact_match/len(data_set), num_len_match/len(data_set), spectrum_time/len(data_set)
  

def decode_beam_select_01(output_top_paths, direction):

  # LAST_LABEL
  if (direction==0):
    LAST_LABEL = data_utils.EOS_ID
  elif (direction==1):
    LAST_LABEL = data_utils.GO_ID

  outputs = []
  
  for entry in xrange(len(output_top_paths)):
    
    top_paths = output_top_paths[entry]

    if not top_paths: # cannot find the peptide
      output_seq = [LAST_LABEL]
      output_score = float("inf")
      #~ print("ERROR: no path found ", entry)
      #~ sys.exit()
    else:
      # path format: [path, score, direction]
      path_scores = np.array([x[1] for x in top_paths])
      top_path = top_paths[np.argmax(path_scores)]
      #
      output_seq = top_path[0][1:]
      output_score = top_path[1]
    
    # output sequence with score
    outputs.append([output_seq,output_score])
  
  return outputs
    

def decode_beam_select_2(output_top_paths):

  outputs = []
  
  for top_paths in output_top_paths:
    
    if not top_paths: # cannot find the peptide
      output_seq = [data_utils.EOS_ID]
      output_score = float("inf")
      #~ print("ERROR: no path found ", entry)
      #~ sys.exit()
    else:
      # path format: [path, score, direction]
      path_scores = np.array([x[1] for x in top_paths])
      path_lengths = np.array([len(x[0]) for x in top_paths])
      #~ top_path = top_paths[np.argmax(path_scores)]
      top_path = top_paths[np.argmax(path_scores/path_lengths)]
      #
      output_seq = top_path[0]
      output_score = top_path[1] / len(output_seq)
      output_seq.append(data_utils.EOS_ID)
      #~ output_score = top_path[1]
    
    # output sequence with score
    outputs.append([output_seq,output_score])
  
  return outputs
    

def decode_beam_search_01(sess, model, knapsack_matrix, 
                          direction, 
                          prefix_mass_list, 
                          precursor_mass_precision, 
                          knapsack_precision, 
                          data_set):

  print("decode_beam_search_01(), direction={0}".format(direction))
  
  # for testing
  test_time_decode = 0.0
  test_time_tf = 0.0
  test_time = 0.0

  # for testing
  start_time_decode = time.time()
  
  # FORWARD/BACKWARD setting
  if (direction==0):
    model_lstm_state0 = model.lstm_state0_forward
    model_output_log_prob = model.output_log_prob_forward
    model_lstm_state = model.lstm_state_forward
    #
    FIRST_LABEL = data_utils.GO_ID
    LAST_LABEL = data_utils.EOS_ID
  #
  elif (direction==1):
    model_lstm_state0 = model.lstm_state0_backward
    model_output_log_prob = model.output_log_prob_backward
    model_lstm_state = model.lstm_state_backward
    #
    FIRST_LABEL = data_utils.EOS_ID
    LAST_LABEL = data_utils.GO_ID






  data_set_len = len(data_set)
  # recall that a data_set[spectrum_id] includes the following
  #     scan                # 0 
  #     spectrum_holder     # 1
  #     spectrum_original   # 2
  #     peptide_mass        # 3
  
  # our TARGET
  output_top_paths = [[] for x in xrange(data_set_len)]
  
  # how many spectra to process at 1 block-run
  decode_block_size = data_utils.batch_size
  





  # for testing
  start_time_tf = time.time()
  #
  # spectrum_holder >> lstm_state0; process in stacks
  data_set_index_list = range(data_set_len)
  data_set_index_stack_list = [data_set_index_list[i:i+decode_block_size] 
                               for i in range(0, data_set_len, decode_block_size)]
  #
  block_c_state0 = []
  block_h_state0 = []
  #
  for stack in data_set_index_stack_list:
    block_spectrum = np.array([data_set[x][1] for x in stack])
    input_feed = {}
    input_feed[model.input_spectrum.name] = block_spectrum
    output_feed = model_lstm_state0
    stack_c_state0, stack_h_state0 = sess.run(fetches=output_feed, feed_dict=input_feed)
    block_c_state0.append(stack_c_state0)
    block_h_state0.append(stack_h_state0)
  #
  block_c_state0 = np.vstack(block_c_state0)
  block_h_state0 = np.vstack(block_h_state0)
  #
  # for testing
  test_time_tf += time.time() - start_time_tf
        





  # hold the spectra & their paths under processing
  active_search = []
  
  # fill in the first entries of active_search
  for spectrum_id in xrange(decode_block_size):
    active_search.append([])
    active_search[-1].append(spectrum_id)
    active_search[-1].append([[[FIRST_LABEL], # current_paths
                                prefix_mass_list[spectrum_id], #data_utils.mass_ID[FIRST_LABEL],
                                0.0,
                                block_c_state0[spectrum_id],
                                block_h_state0[spectrum_id]]])
  
  # how many spectra that have been put into active_search
  spectrum_count = decode_block_size






  # MAIN LOOP; break when the block-run is empty
  while True:
  
    # block-data for model-step-run
    block_AA_ID_1 = [] # nobi
    block_AA_ID_2 = [] # nobi
    block_c_state = []
    block_h_state = []
    block_candidate_intensity = []
    #
    # data to construct new_paths
    block_path_0 = []
    block_prefix_mass = []
    block_score = []
    block_mass_filter_candidate = []
    #
    # store the number of paths of each entry (spectrum) in the big blocks
    entry_block_size = []

    # gather data into blocks
    for entry in active_search:
      
      spectrum_id = entry[0]
      current_paths = entry[1]
      peptide_mass = data_set[spectrum_id][3]
      spectrum_original = data_set[spectrum_id][2]
      
      path_count = 0
  
      for path in current_paths:
  
        AA_ID_2 = path[0][-1] # nobi
        if (len(path[0]) > 1):
          AA_ID_1 = path[0][-2]
        else:
          AA_ID_1 = AA_ID_2 # nobi
        #
        prefix_mass = path[1]
        score = path[2]
        c_state = path[3]
        h_state = path[4]
    
        # reach LAST_LABEL >> check mass
        if (AA_ID_2 == LAST_LABEL): # nobi
          if (abs(prefix_mass - peptide_mass) <= data_utils.PRECURSOR_MASS_PRECISION_TOLERANCE):
            output_top_paths[spectrum_id].append([path[0],path[2],direction])
          continue
    
        # for testing
        start_time = time.time()
      
        # CANDIDATE INTENSITY
        candidate_intensity = get_candidate_intensity(spectrum_original, 
                                                      peptide_mass, 
                                                      prefix_mass, 
                                                      direction)
  
        # for testing
        test_time += time.time() - start_time
      
        # SUFFIX MASS filter
        suffix_mass = peptide_mass - prefix_mass - data_utils.mass_ID[LAST_LABEL]
        mass_filter_candidate = knapsack_search(knapsack_matrix, suffix_mass, knapsack_precision)
        #
        if not mass_filter_candidate: # not enough mass left to extend
          mass_filter_candidate.append(LAST_LABEL) # try to end the sequence
  
        # gather BLOCK-data
        block_AA_ID_1.append(AA_ID_1) # nobi
        block_AA_ID_2.append(AA_ID_2) # nobi
        block_c_state.append(c_state)
        block_h_state.append(h_state)
        block_candidate_intensity.append(candidate_intensity)
        #
        block_path_0.append(path[0])
        block_prefix_mass.append(prefix_mass)
        block_score.append(score)
        block_mass_filter_candidate.append(mass_filter_candidate)
        
        # record the block size for each entry
        path_count += 1
      
      entry_block_size.append(path_count)
              





    # RUN tf blocks if not empty
    if (block_AA_ID_1):

      # for testing
      start_time_tf = time.time()
      #
      # FEED and RUN TensorFlow-model
      block_AA_ID_1 = np.array(block_AA_ID_1) # nobi
      block_AA_ID_2 = np.array(block_AA_ID_2) # nobi
      block_c_state = np.array(block_c_state)
      block_h_state = np.array(block_h_state)
      block_candidate_intensity = np.array(block_candidate_intensity)
      #
      input_feed = {}
      input_feed[model.input_AA_id[0].name] = block_AA_ID_1 # nobi
      input_feed[model.input_AA_id[1].name] = block_AA_ID_2 # nobi
      input_feed[model.input_intensity.name] = block_candidate_intensity
      input_feed[model.input_state[0].name] = block_c_state
      input_feed[model.input_state[1].name] = block_h_state
      #
      #~ output_feed = [model_output_log_prob, model_lstm_state]
      output_feed = model_output_log_prob # nobi
      #
      #~ current_log_prob, (current_c_state, current_h_state) = sess.run(output_feed,input_feed)
      current_log_prob = sess.run(output_feed,input_feed) # nobi
      #
      # for testing
      test_time_tf += time.time() - start_time_tf
    





    # find new_paths for each entry
    block_index = 0
    for entry_index, entry in enumerate(active_search):
      
      new_paths = []

      for index in xrange(block_index, block_index + entry_block_size[entry_index]):
        #
        for aa_id in block_mass_filter_candidate[index]:
          new_paths.append([])
          new_paths[-1].append(block_path_0[index] + [aa_id])
          new_paths[-1].append(block_prefix_mass[index] + data_utils.mass_ID[aa_id])
          #
#~           new_paths[-1].append(block_score[index] + current_log_prob[index][aa_id])
          #
          if (aa_id > 2): # do NOT add score of GO, EOS, PAD
            new_paths[-1].append(block_score[index] + current_log_prob[index][aa_id])
          else:
            new_paths[-1].append(block_score[index])
          #
          #~ new_paths[-1].append(current_c_state[index])
          #~ new_paths[-1].append(current_h_state[index])
          new_paths[-1].append(block_c_state[index]) # nobi
          new_paths[-1].append(block_h_state[index]) # nobi
      
      # pick the top BEAM_SIZE
      if (len(new_paths) > data_utils.FLAGS.beam_size):
        new_path_scores = np.array([x[2] for x in new_paths])
        new_path_lengths = np.array([len(x[0])-1 for x in new_paths])
        top_k_indices = np.argpartition(-new_path_scores,data_utils.FLAGS.beam_size)[:data_utils.FLAGS.beam_size]
        #~ top_k_indices = np.argpartition(-new_path_scores/new_path_lengths,data_utils.FLAGS.beam_size)[:data_utils.FLAGS.beam_size]
        entry[1] = [new_paths[top_k_indices[x]] for x in xrange(data_utils.FLAGS.beam_size)]
      else:
        entry[1] = new_paths[:]
  
      # update the accumulated block_index
      block_index += entry_block_size[entry_index]
    





    # update active_search
    #     by removing empty entries
    active_search = [entry for entry in active_search if entry[1]]
    
    #     and adding new entries
    active_search_len = len(active_search)
    if (active_search_len < decode_block_size and
        spectrum_count < data_set_len):
          
      new_spectrum_count = min(spectrum_count+decode_block_size-active_search_len, data_set_len)

      for spectrum_id in xrange(spectrum_count, new_spectrum_count):
        active_search.append([])
        active_search[-1].append(spectrum_id)
        active_search[-1].append([[[FIRST_LABEL], # current_paths
                                    prefix_mass_list[spectrum_id], #data_utils.mass_ID[FIRST_LABEL],
                                    0.0,
                                    block_c_state0[spectrum_id],
                                    block_h_state0[spectrum_id]]])
        
      spectrum_count = new_spectrum_count
        





    # STOP decoding if no path
    if (not active_search):
      break






  # for testing
  test_time_decode += time.time() - start_time_decode
        
  # for testing
  print("  test_time_tf = %.2f" % (test_time_tf))
  print("  test_time_decode = %.2f" % (test_time_decode))
  print("  test_time = %.2f" % (test_time))
  
  return output_top_paths


def decode_beam_search_2(sess, model, data_set, knapsack_matrix):

  print("decode_beam_search_2()")

  data_set_forward = [[x[0],x[1],x[2],x[4]] for x in data_set] # ignore x[5] (peptide_ids_forward)
  data_set_backward = [[x[0],x[1],x[3],x[4]] for x in data_set] # ignore x[6] (peptide_ids_backward)
  data_set_len = len(data_set_forward)
  peptide_mass_list = [x[4] for x in data_set]
  
  candidate_mass_list = []
  
  # Pick GO mass
  # GO has only one option: prefix_mass
  mass_GO = data_utils.mass_ID[data_utils.GO_ID]
  prefix_mass_list = [mass_GO] * data_set_len
  suffix_mass_list = [(x-mass_GO) for x in peptide_mass_list]
  candidate_mass_list.append([prefix_mass_list, 
                              suffix_mass_list, 
                              0.01, # precursor_mass_precision
                              100 # knapsack_precision
                              ])
  
  # Pick EOS mass
  # EOS has only one option: suffix_mass
  mass_EOS = data_utils.mass_ID[data_utils.EOS_ID]
  prefix_mass_list = [(x-mass_EOS) for x in peptide_mass_list]
  suffix_mass_list = [mass_EOS] * data_set_len
  candidate_mass_list.append([prefix_mass_list, 
                              suffix_mass_list, 
                              0.01, # precursor_mass_precision
                              100 # knapsack_precision
                              ])
  
  # Pick a middle mass
  num_position = data_utils.num_position
  argmax_mass_list = []
  argmax_mass_complement_list = []

  # by choosing the location with max intensity from (0, peptide_mass_C_location)
  for spectrum_id in xrange(data_set_len):

    peptide_mass = peptide_mass_list[spectrum_id]
    peptide_mass_C = peptide_mass - mass_EOS
    peptide_mass_C_location = int(round(peptide_mass_C*data_utils.SPECTRUM_RESOLUTION))
    spectrum_forward = data_set[spectrum_id][2]
    #
    argmax_location = np.argpartition(-spectrum_forward[:peptide_mass_C_location], num_position)[:num_position]
    #
    argmax_mass = argmax_location / data_utils.SPECTRUM_RESOLUTION # !!! LOWER precision 0.1 Da !!!
    #
    # Find the closest theoretical mass to argmax_mass
    #~ aprox_mass = []
    #~ for x in argmax_mass:
      #~ y = knapsack_search_mass(knapsack_matrix, x, 1000) 
      #~ if (not y):
        #~ aprox_mass.append(x)
      #~ else:
        #~ aprox_mass.append(y[0])
    #~ argmax_mass = aprox_mass
    #
    argmax_mass_complement = [(peptide_mass - x) for x in argmax_mass]
    #
    argmax_mass_list.append(argmax_mass) 
    argmax_mass_complement_list.append(argmax_mass_complement)

  # Add the mass and its complement to candidate_mass_list
  for position in xrange(num_position):
    
    prefix_mass_list = [x[position] for x in argmax_mass_list]
    suffix_mass_list = [x[position] for x in argmax_mass_complement_list]
    candidate_mass_list.append([prefix_mass_list, 
                                suffix_mass_list, 
                                0.1, # precursor_mass_precision
                                1000 # knapsack_precision
                                ])
    #
    prefix_mass_list = [x[position] for x in argmax_mass_complement_list]
    suffix_mass_list = [x[position] for x in argmax_mass_list]
    candidate_mass_list.append([prefix_mass_list, 
                                suffix_mass_list, 
                                0.1, # precursor_mass_precision
                                1000 # knapsack_precision
                                ])
    
  # Start decoding for each candidate_mass
  output_top_paths = [[] for x in xrange(data_set_len)]
  for candidate_mass in candidate_mass_list:
    
    top_paths_forward = decode_beam_search_01(sess,
                                              model,
                                              knapsack_matrix,
                                              0,
                                              candidate_mass[0], # prefix_mass_list
                                              candidate_mass[2], # precursor_mass_precision
                                              candidate_mass[3], # knapsack_precision
                                              data_set_forward)

    top_paths_backward = decode_beam_search_01(sess,
                                               model,
                                               knapsack_matrix,
                                               1,
                                               candidate_mass[1], # suffix_mass_list
                                               candidate_mass[2], # precursor_mass_precision
                                               candidate_mass[3], # knapsack_precision
                                               data_set_backward)
                                               
    for spectrum_id in xrange(data_set_len):
                                          
      if (not top_paths_forward[spectrum_id]) or (not top_paths_backward[spectrum_id]): # any list is empty
        continue
      else:
        for x_path in top_paths_forward[spectrum_id]:
          for y_path in top_paths_backward[spectrum_id]:
            
            seq_forward = x_path[0][1:-1]
            seq_backward = y_path[0][1:-1]
            seq_backward = seq_backward[::-1]
            seq = seq_backward + seq_forward
            score = x_path[1] + y_path[1]
            direction = candidate_mass[0][0]
            output_top_paths[spectrum_id].append([seq, score, direction])

  #~ return output_top_paths






  # Refinement using peptide_mass_list, especially for middle mass
  output_top_paths_refined = [[] for x in xrange(data_set_len)]
  for spectrum_id in xrange(data_set_len):
    top_paths = output_top_paths[spectrum_id]
    for path in top_paths:
      seq = path[0]
      seq_mass = sum(data_utils.mass_ID[x] for x in seq)
      seq_mass += mass_GO + mass_EOS
      if abs(seq_mass - peptide_mass_list[spectrum_id]) <= 0.01:
        output_top_paths_refined[spectrum_id].append(path)
      
  
  return output_top_paths_refined
                                                   

def decode_beam_search(sess, model, data_set, knapsack_matrix, output_file_handle):

  print("decode_beam_search()")

  spectrum_time = time.time()

  scans = [x[0] for x in data_set]

  if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
    #
    output_top_paths = decode_beam_search_01(sess, 
                                             model, 
                                             knapsack_matrix,
                                             data_utils.FLAGS.direction,
                                             data_set)
    #
    outputs = decode_beam_select_01(output_top_paths, data_utils.FLAGS.direction)
    #
    spectrum_time = time.time() - spectrum_time
    #
    decoder_inputs = [x[-1] for x in data_set]
    batch_accuracy_AA, batch_len_AA, batch_len_decode, num_exact_match, num_len_match = test_AA_decode_batch(
                                                                                          scans,
                                                                                          decoder_inputs, 
                                                                                          outputs,
                                                                                          data_utils.FLAGS.direction,
                                                                                          output_file_handle)
  else:
    output_top_paths = decode_beam_search_2(sess, 
                                            model, 
                                            data_set,
                                            knapsack_matrix)
    #
    outputs = decode_beam_select_2(output_top_paths)
    #
    spectrum_time = time.time() - spectrum_time
    #
    decoder_inputs_forward = [x[-2] for x in data_set]
    batch_accuracy_AA, batch_len_AA, batch_len_decode, num_exact_match, num_len_match = test_AA_decode_batch(
                                                                                          scans,
                                                                                          decoder_inputs_forward, 
                                                                                          outputs,
                                                                                          0,
                                                                                          output_file_handle)
    

  return batch_accuracy_AA, batch_len_AA, batch_len_decode, num_exact_match, num_len_match, spectrum_time


def decode(input_file=data_utils.decode_test_file):
  
  with tf.Session() as sess:
    
    # DECODING MODEL
    print("DECODING MODEL")
    model = novonet_model.DecodingModel()

#~     test_writer = tf.train.SummaryWriter("test_log", sess.graph)
#~     test_writer.close()
    
    # LOAD PARAMETERS
    print("LOAD PARAMETERS")
    saver = tf.train.Saver(tf.all_variables())
    #
    ckpt = tf.train.get_checkpoint_state(data_utils.FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("ERROR: model parameters not found.")
      sys.exit()

    # FIND SPECTRA LOCATIONS
    spectra_file_location = inspect_file_location(data_format=data_utils.data_format,
                                                  input_file=input_file)
    #
    data_set_len = len(spectra_file_location)
    print("Total number of spectra = {0:d}".format(data_set_len))

    # DECODE with BEAM SEARCH
    if (data_utils.FLAGS.beam_search):

      print("Load knapsack_matrix from default: knapsack.npy")
      knapsack_matrix = np.load("knapsack.npy")
    
      # READ & DECODE in stacks
      print("READ & DECODE in stacks")
      decode_stack_size = data_utils.test_stack_size
      spectra_file_location_stack_list = [spectra_file_location[i:i+decode_stack_size] 
                                          for i in range(0, data_set_len, decode_stack_size)]
    
      total_accuracy_AA = 0.0
      total_len_AA = 0.0
      total_len_decode = 0.0
      total_exact_match = 0.0 
      total_len_match = 0.0 
      total_spectrum_time = 0.0
      total_peptide_decode = 0.0
    
      # print to output file
      decode_output_file = data_utils.FLAGS.train_dir + "/decode_output.tab"
      with open(decode_output_file, 'w') as output_file_handle:

        print("scan\ttarget_seq\toutput_seq\toutput_score\taccuracy_AA\tlen_AA\texact_match\n",
              file=output_file_handle,
              end="")
      
        with open(input_file, 'r') as input_file_handle:
          
          counter_peptide = 0
  
          for stack in spectra_file_location_stack_list:
  
            start_time = time.time()
            stack_data_set, _ = read_spectra(file_handle=input_file_handle, 
                                             data_format=data_utils.data_format, 
                                             spectra_locations=stack)
            counter_peptide += len(stack)
            print("Read {0:d}/{1:d} spectra, reading time = {2:.2f}".format(counter_peptide, 
                                                                            data_set_len,
                                                                            time.time() - start_time))
      
            # concatenate data buckets
            stack_data_set = sum(stack_data_set,[])
      
            batch_accuracy_AA, batch_len_AA, batch_len_decode, num_exact_match, num_len_match, spectrum_time = decode_beam_search(
                                                                                                                 sess, 
                                                                                                                 model, 
                                                                                                                 stack_data_set,
                                                                                                                 knapsack_matrix,
                                                                                                                 output_file_handle)
  
            total_accuracy_AA += batch_accuracy_AA
            total_len_AA += batch_len_AA
            total_len_decode += batch_len_decode
            total_exact_match += num_exact_match 
            total_len_match += num_len_match
            total_spectrum_time += spectrum_time
            total_peptide_decode += len(stack_data_set)
      
      print("ACCURACY SUMMARY")
      print("  recall_AA %.4f" % (total_accuracy_AA/total_len_AA))
      print("  precision_AA %.4f" % (total_accuracy_AA/total_len_decode))
      print("  recall_peptide %.4f" % (total_exact_match/total_peptide_decode))
      print("  recall_len %.4f" % (total_len_match/total_peptide_decode))
      print("  spectrum_time %.4f" % (total_spectrum_time/total_peptide_decode))

    # DECODE with TRUE FEEDING
    else:

      start_time = time.time()
      print("READING SPECTRA")
      with open(input_file, 'r') as file_handle:
        data_set, _ = read_spectra(file_handle=file_handle, 
                                   data_format=data_utils.data_format, 
                                   spectra_locations=spectra_file_location)
      print("Reading time = {0:.2f}".format(time.time() - start_time))

      # decode_true_feeding each bucket separately, like in training/validation
      print("DECODE with TRUE FEEDING")
      for bucket_id in xrange(len(data_utils._buckets)):
        
        if (not data_set[bucket_id]): # empty bucket
          continue

        batch_accuracy_AA, num_exact_match, num_len_match, spectrum_time = decode_true_feeding(
                                                                             sess, 
                                                                             model, 
                                                                             data_set[bucket_id])

        print("ACCURACY SUMMARY - bucket {0}".format(bucket_id))
        print("  accuracy_AA %.4f" % (batch_accuracy_AA))
        print("  accuracy_peptide %.4f" % (num_exact_match))
        print("  accuracy_len %.4f" % (num_len_match))
        print("  spectrum_time %.4f" % (spectrum_time))


def train_cycle(model, sess, 
                input_handle_train, spectra_file_location_train, 
                valid_set, valid_set_len, valid_bucket_len, valid_bucket_pos_id,
                checkpoint_path, log_file_handle, step_time, loss, current_step, best_valid_perplexity):

  # for testing
  run_time = 0.0
  read_time = 0.0
  train_time = 0.0
  train_getBatch_time = 0.0
  train_step_time = 0.0
  valid_time = 0.0
  valid_test_time = 0.0
  valid_save_time = 0.0
  
  # for testing
  run_time_start = time.time()
  
  # for testing
  read_time_start = time.time()

  # Read a RANDOM stack from the train file to train_set
  train_set, _ = read_random_stack(input_handle_train, data_utils.data_format, spectra_file_location_train, 
                                   stack_size=data_utils.train_stack_size)

  # for testing
  read_time = time.time() - read_time_start
  
  #~ print("RESOURCE-train_set: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

  #
  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(data_utils._buckets))]
  train_total_size = float(sum(train_bucket_sizes))
  print("train_bucket_sizes ", train_bucket_sizes)
  print("train_total_size ", train_total_size)
  #
  # for uniform-sample of data from buckets
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]
  print("train_buckets_scale ", train_buckets_scale)
  #
  # to monitor the number of spectra in the current stack
  #   that have been used for training
  train_current_spectra = [0 for b in xrange(len(data_utils._buckets))]
  
########################################################################
  # Get a batch and train
########################################################################
  
  while True:

    start_time = time.time()
    
    # for testing
    train_time_start = time.time()
  
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])
    
    # not enough spectra left in this bucket of the current stack
    # >> break, go up, and load a new RANDOM stack to train_set
    if (train_current_spectra[bucket_id] + data_utils.batch_size >= train_bucket_sizes[bucket_id]):
      print("train_current_spectra ", train_current_spectra)
      print("train_bucket_sizes ", train_bucket_sizes)
      #del train_set # to free memory
      #gc.collect()
      break
    
    # Get a RANDOM batch from the current stack and make a step.
    random_index_list = random.sample(xrange(train_bucket_sizes[bucket_id]), data_utils.batch_size)

    # for testing
    train_getBatch_time_start = time.time()
  
    # get_batch_01/2
    if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
      encoder_inputs, intensity_inputs, decoder_inputs, target_weights = get_batch_01(random_index_list, 
                                                                                      train_set, 
                                                                                      bucket_id)
    else:
      encoder_inputs, intensity_inputs_forward, intensity_inputs_backward, decoder_inputs_forward, decoder_inputs_backward, target_weights = get_batch_2(random_index_list, train_set, bucket_id)

    #~ print("RESOURCE-get_random_batch: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

    # monitor the number of spectra that have been processed
    train_current_spectra[bucket_id] += data_utils.batch_size
    
    # for testing
    train_getBatch_time += time.time() - train_getBatch_time_start
    
    # for testing
    train_step_time_start = time.time()
    
    # model_step
    if (data_utils.FLAGS.direction==0):
      _, step_loss, output_logits = model.step(sess, encoder_inputs, 
                                               intensity_inputs_forward=intensity_inputs,
                                               decoder_inputs_forward=decoder_inputs, 
                                               target_weights=target_weights,
                                               bucket_id=bucket_id, 
                                               training_mode=True)
    elif (data_utils.FLAGS.direction==1):
      _, step_loss, output_logits = model.step(sess, encoder_inputs, 
                                               intensity_inputs_backward=intensity_inputs,
                                               decoder_inputs_backward=decoder_inputs, 
                                               target_weights=target_weights,
                                               bucket_id=bucket_id, 
                                               training_mode=True)
    else:
      _, step_loss, output_logits_forward, output_logits_backward = model.step(sess, encoder_inputs, 
                                                                               intensity_inputs_forward=intensity_inputs_forward,
                                                                               intensity_inputs_backward=intensity_inputs_backward,
                                                                               decoder_inputs_forward=decoder_inputs_forward, 
                                                                               decoder_inputs_backward=decoder_inputs_backward, 
                                                                               target_weights=target_weights,
                                                                               bucket_id=bucket_id, 
                                                                               training_mode=True)

    # for testing
    train_step_time += time.time() - train_step_time_start
    
    #~ print("RESOURCE-step: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

    step_time[0] += (time.time() - start_time) / data_utils.steps_per_checkpoint
    loss[0] += step_loss / data_utils.steps_per_checkpoint
    current_step[0] += 1
    
    # for testing
    train_time += time.time() - train_time_start

########################################################################
    # Checkpoint: Training statistics & Evaluation on valid_set
########################################################################

    if current_step[0] % data_utils.steps_per_checkpoint == 0:

      # for testing
      valid_time_start = time.time()

      # Training statistics for the last round
      #     step_time & loss are averaged over the steps
      #     accuracy is only for the last batch
      perplexity = math.exp(loss[0]) if loss[0] < 300 else float('inf')

      # test_logit_batch_01/2
      if (data_utils.FLAGS.direction==0 or data_utils.FLAGS.direction==1):
        batch_accuracy_AA, batch_len_AA, num_exact_match, _ = test_logit_batch_01(decoder_inputs, 
                                                                                  output_logits)
      else:
        batch_accuracy_AA, batch_len_AA, num_exact_match, _ = test_logit_batch_2(
                                                                decoder_inputs_forward,
                                                                decoder_inputs_backward, 
                                                                output_logits_forward,
                                                                output_logits_backward)

      accuracy_AA = batch_accuracy_AA/batch_len_AA
      accuracy_peptide = num_exact_match/data_utils.batch_size

      epoch = model.global_step.eval()*data_utils.batch_size/len(spectra_file_location_train)
      print ("global step %d epoch %.1f step-time %.2f perplexity %.4f"
             % (model.global_step.eval(), epoch, step_time[0], perplexity))
      print ("%d \t %.1f \t %.2f \t %.4f \t %.4f \t %.4f \t"
             % (model.global_step.eval(), epoch, step_time[0], perplexity, accuracy_AA, accuracy_peptide), 
                file=log_file_handle,
                end="")

      # Zero timer and loss.
      step_time[0], loss[0] = 0.0, 0.0

      # for testing
      valid_test_time_start = time.time()
      
      # Evaluation on valid_set.
      valid_loss = 0.0
      for bucket_id in valid_bucket_pos_id:
        
        #~ eval_ppx, accuracy_AA, accuracy_peptide, _ = test_random_accuracy(sess,
                                                                          #~ model,
                                                                          #~ valid_set, 
                                                                          #~ bucket_id) 
        #
        eval_loss, accuracy_AA, accuracy_peptide, _ = test_accuracy(sess,
                                                                    model,
                                                                    valid_set, 
                                                                    bucket_id,
                                                                    print_summary=False)

        #~ print("RESOURCE-test valid_set: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        
        valid_loss += eval_loss * valid_bucket_len[bucket_id]

        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        print("%d \t %.4f \t %.4f \t %.4f \t" 
              % (bucket_id, eval_ppx, accuracy_AA, accuracy_peptide),
              file=log_file_handle,
              end="")
              
      # for testing
      valid_test_time += time.time() - valid_test_time_start
      
      # for testing
      valid_save_time_start = time.time()

      # Save model with best_valid_perplexity
      valid_loss /= valid_set_len
      valid_ppx = math.exp(valid_loss) if valid_loss < 300 else float('inf')
      if (valid_ppx < best_valid_perplexity[0]):
        best_valid_perplexity[0] = valid_ppx
        #
        model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
        print("best model: valid_ppx %.4f" % (valid_ppx))
  
        #~ print("RESOURCE-saver: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        
      # for testing
      valid_save_time += time.time() - valid_save_time_start

      # for testing
      valid_time += time.time() - valid_time_start

      print("\n", file=log_file_handle, end="")
      log_file_handle.flush()
      
  # for testing
  run_time += time.time() - run_time_start

  # for testing
  print("read_time = {0:.2f}".format(read_time)) 
  print("train_time = {0:.2f}".format(train_time))
  print("   train_getBatch_time = {0:.2f}".format(train_getBatch_time))
  print("   train_step_time = {0:.2f}".format(train_step_time))
  print("valid_time = {0:.2f}".format(valid_time)) 
  print("   valid_test_time = {0:.2f}".format(valid_test_time)) 
  print("   valid_save_time = {0:.2f}".format(valid_save_time)) 
  print("run_time = {0:.2f}".format(run_time)) 


def train():
  print("train()")
  
  #~ print("RESOURCE-train(): ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

  #~ test_inference()
  #~ sys.exit()
  
  # control access to GPU memory
  #~ gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  #~ with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  
  # Inspect train_set, valid_set & record spectra_file_location
  spectra_file_location_train = inspect_file_location(data_utils.data_format, data_utils.input_file_train)
  spectra_file_location_valid = inspect_file_location(data_utils.data_format, data_utils.input_file_valid)

  # read RANDOM stacks to valid_set
  print("read RANDOM stacks to valid_set")
  with open(data_utils.input_file_valid, 'r') as file_handle:
    valid_set, valid_set_len = read_random_stack(file_handle, data_utils.data_format, spectra_file_location_valid, 
                                                 stack_size=data_utils.valid_stack_size)
  valid_bucket_len = [len(x) for x in valid_set]
  valid_bucket_pos_id = np.nonzero(valid_bucket_len)[0]
  
  #~ print("RESOURCE-valid_set and test_set: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)






########################################################################
  # TRAINING on train_set
########################################################################

  with tf.Session() as sess:
    #~ print("RESOURCE-sess: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

    print("Create model for training")
    model = create_model(sess, training_mode=True)

    #~ print("RESOURCE-model: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

    # Open train_set
    input_handle_train = open(data_utils.input_file_train, 'r')
    print("Open train_set: ", data_utils.input_file_train)
    
    # Open log_file
    log_file = data_utils.FLAGS.train_dir + "/log_file_caption_2dir.tab"
    print("Open log_file: ", log_file)
    log_file_handle = open(log_file, 'w')
    print ("global step \t epoch \t step-time \t" 
           "perplexity \t last_accuracy_AA \t last_accuracy_peptide \t" 
           "valid_bucket_id \t perplexity \t accuracy_AA \t accuracy_peptide \n",
            file=log_file_handle,
            end="")

    # Model directory
    checkpoint_path = os.path.join(data_utils.FLAGS.train_dir, "translate.ckpt")
    print("Model directory: ", checkpoint_path)

    # Training record
    best_valid_perplexity = [float("inf")]
    loss = [0.0]
    step_time = [0.0]
    current_step = [0]
    #num_epoch = 0

########################################################################
    # LOOP: Read stacks of spectra and Train
########################################################################

    print("Training loop")

    while True:

      # read RANDOM stacks to valid_set
      #~ print("read RANDOM stacks to valid_set")
      #~ with open(data_utils.input_file_valid, 'r') as file_handle:
        #~ valid_set, valid_set_len = read_random_stack(file_handle, data_utils.data_format, spectra_file_location_valid, 
                                                     #~ stack_size=data_utils.valid_stack_size)
      #~ valid_bucket_len = [len(x) for x in valid_set]
      #~ valid_bucket_pos_id = np.nonzero(valid_bucket_len)[0]
  
      train_cycle(model, 
                  sess, 
                  input_handle_train, 
                  spectra_file_location_train, 
                  valid_set, 
                  valid_set_len,
                  valid_bucket_len,
                  valid_bucket_pos_id,
                  checkpoint_path, 
                  log_file_handle, 
                  step_time, 
                  loss, 
                  current_step, 
                  best_valid_perplexity)

      #~ gc .collect()
      
      print("RESOURCE-train_cycle: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
      
      # stop training if >= 50 epochs
      epoch = model.global_step.eval()*data_utils.batch_size/len(spectra_file_location_train)
      if (epoch >= data_utils.epoch_stop):
        print("EPOCH: {0:.1f}, EXCEED {1:d}, STOP TRAINING LOOP".format(epoch, data_utils.epoch_stop))
        break

    log_file_handle.close()
    
    input_handle_train.close()
    

def test_true_feeding():

  print("test_true_feeding()")
  
  # Inspect file location
  #~ spectra_file_location_valid = inspect_file_location(data_utils.data_format, data_utils.input_file_valid)
  spectra_file_location_test = inspect_file_location(data_utils.data_format, data_utils.input_file_test)

  # Read RANDOM stacks
  #~ with open(data_utils.input_file_valid, 'r') as file_handle:
    #~ valid_set, _ = read_spectra(file_handle, data_utils.data_format, spectra_file_location_valid) 

  with open(data_utils.input_file_test, 'r') as file_handle:
    test_set, _ = read_spectra(file_handle, data_utils.data_format, spectra_file_location_test) 
  
  # Testing
  with tf.Session() as sess:
    #
    print("Create model for testing")
    model = create_model(sess, training_mode=False)
    #
    for bucket_id in xrange(len(data_utils._buckets)):
      
      #~ if (valid_set[bucket_id]): # bucket not empty
        #~ print("valid_set - bucket {0}".format(bucket_id))
        #~ test_accuracy(sess, model, valid_set, bucket_id)
        
      if (test_set[bucket_id]): # bucket not empty
        print("test_set - bucket {0}".format(bucket_id))
        test_accuracy(sess, model, test_set, bucket_id)
    


  
  


