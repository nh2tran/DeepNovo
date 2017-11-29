# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import time
import re
import multiprocessing as mp

from Bio import SeqIO
from pyteomics import parser
import numpy as np
import tensorflow as tf

import deepnovo_config
from deepnovo_cython_modules import get_candidate_intensity


class WorkerDenovo(object):
  """TODO(nh2tran): docstring.
     This class contains the denovo sequencing module.
  """


  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDenovo: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.knapsack_file = deepnovo_config.knapsack_file
    self.MZ_MAX = deepnovo_config.MZ_MAX
    self.mass_N_terminus = deepnovo_config.mass_N_terminus
    self.mass_C_terminus = deepnovo_config.mass_C_terminus
    self.KNAPSACK_AA_RESOLUTION = deepnovo_config.KNAPSACK_AA_RESOLUTION
    self.vocab_size = deepnovo_config.vocab_size
    self.GO_ID = deepnovo_config.GO_ID
    self.EOS_ID = deepnovo_config.EOS_ID
    self.mass_ID = deepnovo_config.mass_ID
    self.precursor_mass_tolerance = deepnovo_config.precursor_mass_tolerance
    self.precursor_mass_ppm = deepnovo_config.precursor_mass_ppm
    self.num_position = deepnovo_config.num_position
    self.SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
    self.mass_AA_min_round = deepnovo_config.mass_AA_min_round
    self.beam_size = deepnovo_config.FLAGS.beam_size
    self.vocab_reverse = deepnovo_config.vocab_reverse
    print("knapsack_file = {0:s}".format(self.knapsack_file))

    # knapsack matrix will be loaded/built at the beginning of search_denovo()
    self.knapsack_matrix = None


  def search_denovo(self, model, worker_io):
    """TODO(nh2tran): docstring.
       Inputs:
         model: tensorflow model, defined in deepnovo_model.ModelInference()
         worker_io: deepnovo_worker_io.WorkerIO() object for input/output tasks
       Outputs:
         predicted_denovo_list: list of predicted peptides, each is a dictionary
           predicted["scan"]
           predicted["sequence"]
           predicted["score"]
           predicted["position_score"]
    """

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDenovo: search_denovo()")

    # output is a list of predicted peptides, each is a dictionary
    #   predicted["scan"]
    #   predicted["sequence"]
    #   predicted["score"]
    #   predicted["position_score"]
    predicted_denovo_list = []

    # load/build knapsack matrix
    if os.path.isfile(self.knapsack_file):
      print("WorkerDenovo: search_denovo() - load knapsack matrix")
      self.knapsack_matrix = np.load(self.knapsack_file)
    else:
      print("WorkerDenovo: search_denovo() - build knapsack matrix")
      self.knapsack_matrix = self._build_knapsack()

    print("WorkerDenovo: search_denovo() - open tensorflow session")
    session = tf.Session()
    model.restore_model(session)

    worker_io.open_input()
    worker_io.get_location()
    worker_io.split_location()
    worker_io.open_output()

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDenovo: search_denovo() - search loop")

    for index, location_batch in enumerate(worker_io.location_batch_list):
      print("Read {0:d}/{1:d} batches".format(index + 1,
                                              worker_io.location_batch_count))
      spectrum_batch = worker_io.get_spectrum(location_batch)
      predicted_batch = self._search_denovo_batch(spectrum_batch, model, session)
      predicted_denovo_list += predicted_batch
      worker_io.write_prediction(predicted_batch)

    print("Total spectra: {0:d}".format(worker_io.spectrum_count["total"]))
    print("  read: {0:d}".format(worker_io.spectrum_count["read"]))
    print("  skipped: {0:d}".format(worker_io.spectrum_count["skipped"]))
    print("    by mass: {0:d}".format(worker_io.spectrum_count["skipped_mass"]))

    worker_io.close_input()
    worker_io.close_output()
    session.close()

    return predicted_denovo_list


  def _build_knapsack(self):
    """TODO(nh2tran): docstring.
       Build a static knapsack matrix by using dynamic programming.
       The knapsack matrix allows to retrieve all possible amino acids that
       could sum up to a given mass, subject to a given resolution.
    """

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDenovo: _build_knapsack()")

    # maximum peptide mass, adjusted by the two terminals
    max_mass = self.MZ_MAX
    max_mass -= self.mass_N_terminus + self.mass_C_terminus
    # convert from float to integer as the algorithm only works with integer
    max_mass_round = int(round(max_mass * self.KNAPSACK_AA_RESOLUTION))
    # allow error tolerance up to 1 Dalton
    max_mass_upperbound = max_mass_round + self.KNAPSACK_AA_RESOLUTION

    knapsack_matrix = np.zeros(shape=(self.vocab_size, max_mass_upperbound),
                               dtype=bool)

    # fill up the knapsack_matrix by rows and columns, using dynamic programming
    for AAid in xrange(3, self.vocab_size): # excluding PAD, GO, EOS

      mass_AA = int(round(self.mass_ID[AAid] * self.KNAPSACK_AA_RESOLUTION))

      for col in xrange(max_mass_upperbound):
  
        # col 0 ~ mass 1
        # col + 1 = mass
        # col = mass - 1
        current_mass = col + 1
  
        if current_mass < mass_AA:
          knapsack_matrix[AAid, col] = False

        elif current_mass == mass_AA:
          knapsack_matrix[AAid, col] = True

        elif current_mass > mass_AA:
          sub_mass = current_mass - mass_AA
          sub_col = sub_mass - 1
          # check if the sub_mass can be formed by a combination of amino acids
          # TODO(nh2tran): change np.sum to np.any
          if np.sum(knapsack_matrix[:, sub_col]) > 0:
            knapsack_matrix[AAid, col] = True
            knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col],
                                                    knapsack_matrix[:, sub_col])
          else:
            knapsack_matrix[AAid, col] = False
  
    np.save(self.knapsack_file, knapsack_matrix)
    return knapsack_matrix


  def _extend_peak(self,
                   direction,
                   session,
                   model,
                   spectrum_batch,
                   peak_batch):
    """TODO(nh2tran): docstring.
       Inputs:
         spectrum_batch: a list of spectrum, each is a dictionary
           spectrum["scan"]
           spectrum["precursor_mass"]
           spectrum["spectrum_holder"]
           spectrum["spectrum_original_forward"]
           spectrum["spectrum_original_backward"]
         peak_batch: one peak for each spectrum, each peak is a dictionary
           peak["prefix_mass"] for extension in the forward direction
           peak["sufffix_mass"] for extension in the backward direction
           peak["mass_tolerance"]
       Outputs:
         top_path_batch: for every input spectrum, the output is a list of paths,
           each path is a dictionary
             path["AAid_list"]
             path["score_list"]
             path["score_sum"]
    """

    print("WorkerDenovo: _extend_peak(), direction={0:s}".format(direction))

    # test running time and tensorflow time
    test_time_decode = 0.0
    test_time_tf = 0.0
    test_time = 0.0
    start_time_decode = time.time()

    # for every input spectrum, the output is a list of paths,
    #   each path is a dictionary
    #   path["AAid_list"]
    #   path["score_list"]
    #   path["score_sum"]
    spectrum_batch_size = len(spectrum_batch)
    top_path_batch = [[] for x in xrange(spectrum_batch_size)]

    # forward/backward direction setting
    #   the direction determines the model, the spectrum and the peak mass
    if direction == "forward":
      model_lstm_state0 = model.output_forward["lstm_state0"]
      model_output_log_prob = model.output_forward["logprob"]
      model_lstm_state = model.output_forward["lstm_state"]
      spectrum_original_name = "spectrum_original_forward"
      peak_mass_name = "prefix_mass"
      FIRST_LABEL = self.GO_ID
      LAST_LABEL = self.EOS_ID
    elif direction == "backward":
      model_lstm_state0 = model.output_backward["lstm_state0"]
      model_output_log_prob = model.output_backward["logprob"]
      model_lstm_state = model.output_backward["lstm_state"]
      spectrum_original_name = "spectrum_original_backward"
      peak_mass_name = "suffix_mass"
      FIRST_LABEL = self.EOS_ID
      LAST_LABEL = self.GO_ID

    # PEAK EXTENSION includes 4 steps:
    #   STEP 1: initialize the lstm and the active_search_list.
    #   STEP 2, 3, 4 are repeated until the active_search_list is empty.
    #     STEP 2: gather data from active search entries and group into blocks.
    #     STEP 3: run tensorflow model on data blocks to predict next AA.
    #     STEP 4: retrieve data from blocks to update the active_search_list
    #       with knapsack dynamic programming and beam search.

    start_time_tf = time.time()
    # STEP 1: initialize lstm
    spectrum_holder_array = np.array([x["spectrum_holder"] for x in spectrum_batch])
    input_feed = {}
    input_feed[model.input_dict["spectrum"].name] = spectrum_holder_array
    output_feed = model_lstm_state0
    c_state0_array, h_state0_array = session.run(fetches=output_feed,
                                                 feed_dict=input_feed)
    test_time_tf += time.time() - start_time_tf

    # STEP 1: initialize the active_search_list
    # active_search_list holds the info of search entries under processing
    #   each search entry is a dictionary
    #     search_entry["spectrum_id"]
    #     search_entry["current_path_list"]
    #   each path is also a dictionary
    #     path["AAid_list"]
    #     path["prefix_mass"]
    #     path["score_list"]
    #     path["score_sum"]
    #     path["c_state"]
    #     path["h_state"]
    active_search_list = []
    for spectrum_id in xrange(spectrum_batch_size):
      search_entry = {}
      search_entry["spectrum_id"] = spectrum_id
      path = {}
      path["AAid_list"] = [FIRST_LABEL]
      path["prefix_mass"] = peak_batch[spectrum_id][peak_mass_name]
      path["score_list"] = [0.0]
      path["score_sum"] = 0.0
      path["c_state"] = c_state0_array[spectrum_id]
      path["h_state"] = h_state0_array[spectrum_id]
      search_entry["current_path_list"] = [path]
      active_search_list.append(search_entry)

    # repeat STEP 2, 3, 4 until the active_search_list is empty.
    while True:

      # STEP 2: gather data from active search entries and group into blocks.

      # data blocks for the input feed of tensorflow model
      block_AAid_1 = [] # nobi
      block_AAid_2 = [] # nobi
      block_c_state = []
      block_h_state = []
      block_candidate_intensity = []
      # data blocks to record the current status of search entries
      block_AAid_list = []
      block_prefix_mass = []
      block_score_list = []
      block_score_sum = []
      block_knapsack_candidate = []

      # store the number of paths of each search entry in the big blocks
      #   to retrieve the info of each search entry later in STEP 4.
      search_entry_size = [0] * len(active_search_list)

      # gather data into blocks through 2 nested loops over active_search_list
      #   and over current_path_list of each search_entry
      for entry_index, search_entry in enumerate(active_search_list):

        spectrum_id = search_entry["spectrum_id"]
        current_path_list = search_entry["current_path_list"]
        precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
        spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name]
        peak_mass_tolerance = peak_batch[spectrum_id]["mass_tolerance"]

        for path in current_path_list:

          # keep track of the AA predicted in the previous iteration
          #   for nobi (short k-mer) model, we will need 2 previous AA
          AAid_list = path["AAid_list"]
          AAid_2 = AAid_list[-1]
          if len(AAid_list) > 1:
            AAid_1 = AAid_list[-2]
          else:
            AAid_1 = AAid_2 # nobi

          # the current status of this path
          prefix_mass = path["prefix_mass"]
          score_list = path["score_list"]
          score_sum = path["score_sum"]
          c_state = path["c_state"]
          h_state = path["h_state"]

          # when we reach LAST_LABEL, check if the mass of predicted sequence
          #   is close to the given precursor_mass:
          #   if yes, send the current path to output
          #   if not, skip the current path
          if AAid_2 == LAST_LABEL: # nobi
            if (abs(prefix_mass - precursor_mass) <= peak_mass_tolerance):
              top_path_batch[spectrum_id].append({"AAid_list": AAid_list,
                                                  "score_list": score_list,
                                                  "score_sum": score_sum})
            continue

          start_time = time.time()
          # get CANDIDATE INTENSITY to predict next AA
          # TODO(nh2tran): change direction from 0/1 to "forward"/"backward"
          direction_id = 0 if direction=="forward" else 1
          candidate_intensity = get_candidate_intensity(spectrum_original,
                                                        precursor_mass,
                                                        prefix_mass,
                                                        direction_id)
          test_time += time.time() - start_time

          # use knapsack and SUFFIX MASS to filter next AA candidate
          suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL]
          knapsack_tolerance = int(round(peak_mass_tolerance
                                         * self.KNAPSACK_AA_RESOLUTION))
          knapsack_candidate = self._search_knapsack(suffix_mass,
                                                     knapsack_tolerance)
          # if not possible to extend, add LAST_LABEL to end the sequence
          if not knapsack_candidate:
            knapsack_candidate.append(LAST_LABEL)

          # gather data blocks
          block_AAid_1.append(AAid_1) # nobi
          block_AAid_2.append(AAid_2) # nobi
          block_c_state.append(c_state)
          block_h_state.append(h_state)
          block_candidate_intensity.append(candidate_intensity)
  
          block_AAid_list.append(AAid_list)
          block_prefix_mass.append(prefix_mass)
          block_score_list.append(score_list)
          block_score_sum.append(score_sum)
          block_knapsack_candidate.append(knapsack_candidate)
  
          # record the size of each search entry in the blocks
          search_entry_size[entry_index] += 1

      # STEP 3: run tensorflow model on data blocks to predict next AA.
      #   output is stored in current_log_prob, current_c_state, current_h_state
      if block_AAid_1:

        start_time_tf = time.time()

        block_AAid_1 = np.array(block_AAid_1) # nobi
        block_AAid_2 = np.array(block_AAid_2) # nobi
        block_c_state = np.array(block_c_state)
        block_h_state = np.array(block_h_state)
        block_candidate_intensity = np.array(block_candidate_intensity)

        input_feed = {}
        input_feed[model.input_dict["AAid"][0].name] = block_AAid_1 # nobi
        input_feed[model.input_dict["AAid"][1].name] = block_AAid_2 # nobi
        input_feed[model.input_dict["lstm_state"][0].name] = block_c_state
        input_feed[model.input_dict["lstm_state"][1].name] = block_h_state
        input_feed[model.input_dict["intensity"].name] = block_candidate_intensity

        output_feed = [model_output_log_prob, model_lstm_state] # lstm.len_full
        #~ output_feed = model_output_log_prob # nobi

        current_log_prob, (current_c_state, current_h_state) = session.run(
            output_feed,
            input_feed) # lstm.len_full
        #~ current_log_prob = session.run(output_feed,input_feed) # nobi

        test_time_tf += time.time() - start_time_tf

      # STEP 4: retrieve data from blocks to update the active_search_list
      #   with knapsack dynamic programming and beam search.
      block_index = 0
      for entry_index, search_entry in enumerate(active_search_list):

        # find all possible new paths within knapsack filter
        new_path_list = []
        for index in xrange(block_index, block_index + search_entry_size[entry_index]):
          for AAid in block_knapsack_candidate[index]:
            new_path = {}
            new_path["AAid_list"] = block_AAid_list[index] + [AAid]
            new_path["prefix_mass"] = block_prefix_mass[index] + self.mass_ID[AAid]
            if AAid > 2: # do NOT add score of GO, EOS, PAD
              new_path["score_list"] = (block_score_list[index]
                                        + [current_log_prob[index][AAid]])
              new_path["score_sum"] = (block_score_sum[index]
                                       + current_log_prob[index][AAid])
            else:
              new_path["score_list"] = block_score_list[index]
              new_path["score_sum"] = block_score_sum[index]
            new_path["c_state"] = current_c_state[index] # lstm.len_full
            new_path["h_state"] = current_h_state[index] # lstm.len_full
            #~ new_path["c_state"] = block_c_state[index] # nobi
            #~ new_path["h_state"] = block_h_state[index] # nobi
            new_path_list.append(new_path)
  
        # beam search to select top candidates
        if len(new_path_list) > self.beam_size:
          new_path_score = np.array([x["score_sum"] for x in new_path_list])
          top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size] # pylint: disable=line-too-long
          search_entry["current_path_list"] = [new_path_list[top_k_index[x]]
                                               for x in xrange(self.beam_size)]
        else:
          search_entry["current_path_list"] = new_path_list
  
        # update the accumulated block_index
        block_index += search_entry_size[entry_index]
  
      # update active_search_list by removing empty entries
      active_search_list = [x for x in active_search_list if x["current_path_list"]]
      # STOP the extension loop if active_search_list is empty
      if not active_search_list:
        break

    test_time_decode += time.time() - start_time_decode
    print("  test_time_tf = %.2f" % (test_time_tf))
    print("  test_time_decode = %.2f" % (test_time_decode))
    print("  test_time = %.2f" % (test_time))
  
    return top_path_batch


  def _search_denovo_batch(self, spectrum_batch, model, session):
    """TODO(nh2tran): docstring.
       Inputs:
         spectrum_batch: a list of spectrum, each is a dictionary
           spectrum["scan"]
           spectrum["precursor_mass"]
           spectrum["spectrum_holder"]
           spectrum["spectrum_original_forward"]
           spectrum["spectrum_original_backward"]
       Outputs:
         predicted_batch: a list of predicted, each is a dictionary
           predicted["scan"]
           predicted["sequence"]
           predicted["score"]
           predicted["position_score"]
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDenovo: _search_denovo_batch()")

    spectrum_batch_size = len(spectrum_batch)

    # direction=2 by default, rarely used
    #output_top_paths = decode_beam_search_2(sess, model, data_set, knapsack_matrix)

    # select peaks for forward/backward/middle extension
    peak_list = self._select_peak(spectrum_batch)

    # extend peaks for each spectrum and record candidates in top_candidate_batch
    #   each spectrum has a list of candidates, each candidate is a dictionary
    #   candidate["sequence"]
    #   candidate["position_score"]
    #   candidate["score"]
    top_candidate_batch = [[] for x in xrange(spectrum_batch_size)]
    for peak_batch in peak_list:

      forward_path_batch = self._extend_peak("forward",
                                             session,
                                             model,
                                             spectrum_batch,
                                             peak_batch)
      backward_path_batch = self._extend_peak("backward",
                                              session,
                                              model,
                                              spectrum_batch,
                                              peak_batch)

      # concatenate forward and backward paths
      for spectrum_id in xrange(spectrum_batch_size):
        if ((not forward_path_batch[spectrum_id])
            or (not backward_path_batch[spectrum_id])): # any list is empty
          continue
        else:
          for x_path in forward_path_batch[spectrum_id]:
            for y_path in backward_path_batch[spectrum_id]:
              AAid_list_forward = x_path["AAid_list"][1:-1]
              score_list_forward = x_path["score_list"][1:-1]
              score_sum_forward = x_path["score_sum"]
              AAid_list_backward = y_path["AAid_list"][1:-1]
              score_list_backward = y_path["score_list"][1:-1]
              score_sum_backward = y_path["score_sum"]
              # reverse backward lists
              AAid_list_backward = AAid_list_backward[::-1]
              score_list_backward = score_list_backward[::-1]
              # concatenate backward and forward lists
              sequence = AAid_list_backward + AAid_list_forward
              position_score = score_list_backward + score_list_forward
              score = score_sum_backward + score_sum_forward
              top_candidate_batch[spectrum_id].append({
                  "sequence": sequence,
                  "position_score": position_score,
                  "score": score})

    # refine and select the best sequence for each spectrum
    predicted_batch = self._select_sequence(spectrum_batch, top_candidate_batch)

    return predicted_batch


  def _search_knapsack(self, mass, knapsack_tolerance):
    """TODO(nh2tran): docstring.
       Given a mass and a tolerance, return the list of candidate AAid.
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDenovo: _search_denovo_batch()")

    # convert the mass and tolerance to a range of columns of knapsack_matrix
    mass_round = int(round(mass * self.KNAPSACK_AA_RESOLUTION))
    mass_upperbound = mass_round + knapsack_tolerance
    mass_lowerbound = mass_round - knapsack_tolerance
  
    # [mass_lowerbound, mass_upperbound] will NOT be less than
    #   mass_AA_min_round.
    if mass_upperbound < self.mass_AA_min_round: # 57.0215
      return []
    # mass_upperbound may exceed column 2982.9895,
    #   but numpy will ignore the extra indices.
    # not necessary, because mass_upperbound > 57.0215
    #~ if (mass_lowerbound < 0):
      #~ return []

    # col 0 ~ mass 1
    # col + 1 = mass
    # col = mass - 1
    # [)
    mass_lowerbound_col = mass_lowerbound - 1
    mass_upperbound_col = mass_upperbound - 1
    # Search for any nonzero col
    candidate_AAid = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:mass_upperbound_col+1], # pylint: disable=line-too-long
                                            axis=1))
  
    return candidate_AAid.tolist()


  def _select_peak(self, spectrum_batch):
    """TODO(nh2tran): docstring.
       Select a given number of peaks for each spectrum for extension.
       Inputs:
         spectrum_batch: a list of spectrum, each is a dictionary
           spectrum["scan"]
           spectrum["precursor_mass"]
           spectrum["spectrum_holder"]
           spectrum["spectrum_original_forward"]
           spectrum["spectrum_original_backward"]
       Outputs:
         peak_list: a list of peak_batch,
           each peak_batch contains 1 peak for every spectrum in spectrum_batch,
           each peak is a dictionary
           peak["prefix_mass"] for extension in the forward direction
           peak["sufffix_mass"] for extension in the backward direction
           peak["mass_tolerance"]
    """

    peak_list = []
    spectrum_batch_size = len(spectrum_batch)

    # select GO peak, GO only corresponds to prefix_mass
    mass_GO = self.mass_ID[self.GO_ID]
    peak_batch = [{"prefix_mass": mass_GO,
                   "suffix_mass": x["precursor_mass"] - mass_GO,
                   "mass_tolerance": self.precursor_mass_tolerance}
                  for x in spectrum_batch]
    peak_list.append(peak_batch)

    # select EOS peak, EOS only corresponds to suffix_mass
    mass_EOS = self.mass_ID[self.EOS_ID]
    peak_batch = [{"prefix_mass": x["precursor_mass"] - mass_EOS,
                   "suffix_mass": mass_EOS,
                   "mass_tolerance": self.precursor_mass_tolerance}
                  for x in spectrum_batch]
    peak_list.append(peak_batch)

    # select a number of middle peaks by choosing the location of max intensity
    #   from (0, precursor_mass_C_location) of each spectrum
    argmax_mass_batch = []
    argmax_mass_complement_batch = []
    for spectrum in spectrum_batch:
      precursor_mass = spectrum["precursor_mass"]
      precursor_mass_C = precursor_mass - mass_EOS
      precursor_mass_C_location = int(round(precursor_mass_C
                                          * self.SPECTRUM_RESOLUTION))
      spectrum_forward = spectrum["spectrum_original_forward"]
      argmax_location = np.argpartition(-spectrum_forward[:precursor_mass_C_location], self.num_position)[:self.num_position] # pylint: disable=line-too-long
      # NOTE that the precursor mass tolerance from now on should depend on
      #   SPECTRUM_RESOLUTION i.e. ms2 tolerance
      argmax_mass = argmax_location / self.SPECTRUM_RESOLUTION
      argmax_mass_complement = [(precursor_mass - x) for x in argmax_mass]
      argmax_mass_batch.append(argmax_mass)
      argmax_mass_complement_batch.append(argmax_mass_complement)

    # NOTE that the peak mass tolerance now depends on SPECTRUM_RESOLUTION,
    #   because the peak was selected from the ms2 spectrum
    mass_tolerance = 1./self.SPECTRUM_RESOLUTION

    # add middle peaks and their complements to peak_list
    for index in xrange(self.num_position):

      # treat the peak as a b-ion, so it corresponds to a prefix, and its
      #   complement y-ion corresponds to a suffix
      peak_batch = [{"prefix_mass": b[index],
                     "suffix_mass": y[index],
                     "mass_tolerance": mass_tolerance}
                    for b, y in zip(argmax_mass_batch,
                                    argmax_mass_complement_batch)]
      peak_list.append(peak_batch)

      # treat the peak as a y-ion, so it corresponds to a suffix, and its
      #   complement b-ion corresponds to a prefix
      peak_batch = [{"prefix_mass": b[index],
                     "suffix_mass": y[index],
                     "mass_tolerance": mass_tolerance}
                    for b, y in zip(argmax_mass_complement_batch,
                                    argmax_mass_batch)]
      peak_list.append(peak_batch)

    return peak_list


  def _select_sequence(self, spectrum_batch, top_candidate_batch):
    """TODO(nh2tran): docstring.
       Inputs:
         spectrum_batch: a list of spectrum, each is a dictionary
           spectrum["scan"]
           spectrum["precursor_mass"]
           spectrum["spectrum_holder"]
           spectrum["spectrum_original_forward"]
           spectrum["spectrum_original_backward"]
       Outputs:
         predicted_batch: a list of predicted, each is a dictionary
           predicted["scan"]
           predicted["sequence"]
           predicted["score"]
           predicted["position_score"]
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDenovo: _select_sequence()")

    spectrum_batch_size = len(spectrum_batch)

    # refine/filter predicted sequences by precursor mass,
    #   especially for middle peak extension
    refine_batch = [[] for x in xrange(spectrum_batch_size)]
    for spectrum_id in xrange(spectrum_batch_size):
      precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
      candidate_list = top_candidate_batch[spectrum_id]
      for candidate in candidate_list:
        sequence = candidate["sequence"]
        sequence_mass = sum(self.mass_ID[x] for x in sequence)
        sequence_mass += self.mass_ID[self.GO_ID] + self.mass_ID[self.EOS_ID]
        if abs(sequence_mass - precursor_mass) <= self.precursor_mass_tolerance:
          refine_batch[spectrum_id].append(candidate)

    # select the best len-normalized scoring candidate
    predicted_batch = [[] for x in xrange(spectrum_batch_size)]
    for spectrum_id in xrange(spectrum_batch_size):

      predicted_batch[spectrum_id] = {}
      predicted_batch[spectrum_id]["scan"] = spectrum_batch[spectrum_id]["scan"]

      candidate_list = refine_batch[spectrum_id]
      if not candidate_list: # cannot find any peptide
        predicted_batch[spectrum_id]["sequence"] = []
        predicted_batch[spectrum_id]["position_score"] = []
        predicted_batch[spectrum_id]["score"] = -float("inf")
      else:
        score_array = np.array([x["score"] for x in candidate_list])
        len_array = np.array([len(x["sequence"]) for x in candidate_list])
        predicted = candidate_list[np.argmax(score_array/len_array)]
        predicted_batch[spectrum_id]["score"] = predicted["score"]
        predicted_batch[spectrum_id]["score"] /= len(predicted["sequence"])
        predicted_batch[spectrum_id]["position_score"] = predicted["position_score"]
        # NOTE that we convert AAid back to letter
        predicted_batch[spectrum_id]["sequence"] = [self.vocab_reverse[x]
                                                    for x in predicted["sequence"]]

    return predicted_batch


