# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import re
import multiprocessing as mp

from Bio import SeqIO
from pyteomics import parser
import numpy as np
import tensorflow as tf

import deepnovo_config
from deepnovo_cython_modules import get_candidate_intensity


class WorkerDB(object):
  """TODO(nh2tran): docstring.
     We use "db" for "database".
     We use "pepmod" to refer to a modified version of a "peptide"
  """


  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB.__init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    # input info to build a db
    self.db_fasta_file = deepnovo_config.db_fasta_file
    self.cleavage_rule = deepnovo_config.cleavage_rule
    self.num_missed_cleavage = deepnovo_config.num_missed_cleavage
    self.fixed_mod_list = deepnovo_config.fixed_mod_list
    self.var_mod_list = deepnovo_config.var_mod_list
    self.mass_tolerance = deepnovo_config.mass_tolerance
    self.ppm = deepnovo_config.ppm
    print("db_fasta_file = {0:s}".format(self.db_fasta_file))
    print("cleavage_rule = {0:s}".format(self.cleavage_rule))
    print("num_missed_cleavage = {0:d}".format(self.num_missed_cleavage))
    print("fixed_mod_list = {0}".format(self.fixed_mod_list))
    print("var_mod_list = {0}".format(self.var_mod_list))
    print("mass_tolerance = {0:.4f}".format(self.mass_tolerance))
    print("ppm = {0:.6f}".format(self.ppm))

    # data structure to store a db
    # all attributes will be built/loaded by build_db()
    self.peptide_count = None
    self.peptide_list = None
    self.peptide_mass_array = None
    self.pepmod_maxmass_array = None


  def build_db(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB.build_db()")

    # parse the input fasta file into a list of sequences
    # more about SeqIO and SeqRecord: http://biopython.org/wiki/SeqRecord
    with open(self.db_fasta_file, "r") as handle:
      record_iterator = SeqIO.parse(handle, "fasta")
      record_list = list(record_iterator)
      sequence_list = [str(record.seq) for record in record_list]
      print("Number of protein sequences: {0:d}".format(len(sequence_list)))

    # cleave protein sequences into a list of unique peptides
    # more about pyteomics.parser.cleave and cleavage rules:
    # https://pythonhosted.org/pyteomics/api/parser.html
    peptide_set = set()
    for sequence in sequence_list:
      peptide_set.update((parser.cleave(
          sequence=sequence,
          rule=parser.expasy_rules[self.cleavage_rule],
          missed_cleavages=self.num_missed_cleavage)))
    peptide_list = list(peptide_set)

    # skip peptides with undetermined amino acid 'X', or 'B'
    peptide_list = [list(peptide) for peptide in peptide_list
                    if not ('X' in peptide or 'B' in peptide)]
    peptide_count = len(peptide_list)
    print("Number of peptides: {0:d}".format(peptide_count))

    # replace "L" by "I"
    for index, peptide in enumerate(peptide_list):
      peptide = ['I' if x == 'L' else x for x in peptide]
      peptide_list[index] = peptide

    # update fixed modifications
    for index, peptide in enumerate(peptide_list):
      peptide = [x + 'mod' if x in self.fixed_mod_list else x for x in peptide]
      peptide_list[index] = peptide

    # for each peptide, find the mass and the max modification mass
    peptide_mass_array = np.zeros(peptide_count)
    pepmod_maxmass_array = np.zeros(peptide_count)
    for index, peptide in enumerate(peptide_list):
      peptide_mass_array[index] = self._compute_peptide_mass(peptide)
      pepmod = [x + 'mod' if x in self.var_mod_list else x for x in peptide]
      pepmod_maxmass_array[index] = self._compute_peptide_mass(pepmod)

    self.peptide_count = peptide_count
    self.peptide_list = peptide_list
    self.peptide_mass_array = peptide_mass_array
    self.pepmod_maxmass_array = pepmod_maxmass_array


  def search_db(self, model, worker_io):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB.search_db()")

    print("WorkerDB.search_db() - open tensorflow session")
    session = tf.Session()
    model.restore(session)

    worker_io.open_input()
    worker_io.get_location()
    worker_io.split_location()
    worker_io.open_output()

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB.search_db() - search loop")

    for index, location_batch in enumerate(worker_io.location_batch_list):
      print("Read {0:d}/{1:d} batches".format(index + 1,
                                              worker_io.location_batch_count))
      spectrum_batch = worker_io.get_spectrum(location_batch)
      predicted_batch = self._search_db_batch(spectrum_batch, model, session)
      worker_io.write_prediction(predicted_batch)

    print("Total spectra: {0:d}".format(worker_io.spectrum_count["total"]))
    print("  read: {0:d}".format(worker_io.spectrum_count["read"]))
    print("  skipped: {0:d}".format(worker_io.spectrum_count["skipped"]))
    print("    by mass: {0:d}".format(worker_io.spectrum_count["skipped_mass"]))

    worker_io.close_input()
    worker_io.close_output()
    session.close()


  def _search_db_batch(self, spectrum_batch, model, session):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB._search_db_batch()")

    # initialize the lstm using the spectrum
    # for faster speed, we initialize the whole spectrum_batch instead of 1-by-1
    input_feed = {}
    spectrum_holder = np.array([spectrum["spectrum_holder"]
                                for spectrum in spectrum_batch])
    input_feed[model.input_spectrum.name] = spectrum_holder
    output_feed = [model.lstm_state0_forward, model.lstm_state0_backward]
    ((state0_c_forward, state0_h_forward),
     (state0_c_backward, state0_h_backward)) = session.run(fetches=output_feed,
                                                           feed_dict=input_feed)

    predicted_batch = []
    # we search spectrum by spectrum
    # a faster way is to process them in parallel, but hard to debug
    for spectrum_index, spectrum in enumerate(spectrum_batch):

      # filter by precursor mass
      # example: [['M', 'D', 'K', 'F', 'Nmod', 'K', 'K']]
      candidate_list = self._filter_by_mass(spectrum["precursor_mass"])

      # add special GO/EOS and reverse
      # example: [['_GO', 'M', 'D', 'K', 'F', 'Nmod', 'K', 'K', '_EOS']]
      candidate_forward_list = [[deepnovo_config._GO] + x + [deepnovo_config._EOS]
                                for x in candidate_list]
      candidate_backward_list = [x[::-1] for x in candidate_forward_list]

      # add PAD to all candidates to the same max length
      # [['_GO', 'M', 'D', 'K', 'F', 'Nmod', 'K', 'K', '_EOS', '_PAD', '_PAD']]
      # due to the same precursor mass, candidates have very similar lengths
      candidate_len_list = [len(x) for x in candidate_list]
      candidate_maxlen = max(candidate_len_list)
      for index, length in enumerate(candidate_len_list):
        if length < candidate_maxlen:
          pad_size = candidate_maxlen - length
          candidate_forward_list[index] += [deepnovo_config._PAD] * pad_size
          candidate_backward_list[index] += [deepnovo_config._PAD] * pad_size
      
      # score the spectrum against its candidates
      #   using the forward model
      logprob_forward_list = self._score_spectrum(
          spectrum["precursor_mass"],
          spectrum["spectrum_original_forward"],
          state0_c_forward[spectrum_index],
          state0_h_forward[spectrum_index],
          candidate_forward_list,
          model,
          model.output_log_prob_forward,
          model.lstm_state_forward,
          session,
          direction=0)
      #   and using the backward model
      logprob_backward_list = self._score_spectrum(
          spectrum["precursor_mass"],
          spectrum["spectrum_original_backward"],
          state0_c_backward[spectrum_index],
          state0_h_backward[spectrum_index],
          candidate_backward_list,
          model,
          model.output_log_prob_backward,
          model.lstm_state_backward,
          session,
          direction=1)

      # note that the candidates are grouped into minibatches
      # === candidate_len ===
      # s
      # i
      # z
      # e
      # =====================
      # logprob_forward_list is a list of candidate_maxlen arrays of shape
      #   [minibatch_size, 26]
      # each row is log of probability distribution over 26 classes/symbols

      # find the best scoring candidate
      predicted = {"scan": "",
                   "sequence": [],
                   "score": -float("inf"),
                   "position_score": []}

      for index, candidate in enumerate(candidate_list):

        # only calculate score on the actual length, not on GO/EOS/PAD
        candidate_len = candidate_len_list[index]

        # align forward and backward logprob
        logprob_forward = [logprob_forward_list[position][index]
                           for position in range(candidate_len)]
        logprob_backward = [logprob_backward_list[position][index]
                            for position in range(candidate_len)]
        logprob_backward = logprob_backward[::-1]

        # score is the sum of logprob(AA) of the candidate in both directions
        #   averaged by the candidate length
        position_score = []
        for position in range(candidate_len):
          AA = candidate[position]
          AA_id = deepnovo_config.vocab[AA]
          position_score.append(logprob_forward[position][AA_id]
                                + logprob_backward[position][AA_id])
        score = sum(position_score) / candidate_len
        if score > predicted["score"]:
          predicted["scan"] = spectrum["scan"]
          predicted["sequence"] = candidate
          predicted["score"] = score
          predicted["position_score"] = position_score
        #~ if (spectrum["scan"]=="F1:11201"):
          #~ print(score, candidate)
          #~ print(spectrum["precursor_mass"])
          #~ print(self._compute_peptide_mass(['Y', 'Y', 'G', 'G', 'N', 'E', 'H', 'I', 'D', 'R']))
          #~ print(self._compute_peptide_mass(['A', 'A', 'E', 'E', 'N', 'F', 'N', 'A', 'D', 'D', 'K']))
      #~ if (spectrum["scan"]=="F1:11201"):
        #~ sys.exit()
      predicted_batch.append(predicted)

    return predicted_batch


  def _score_spectrum(self,
                      precursor_mass,
                      spectrum_original,
                      state0_c,
                      state0_h,
                      candidate_list,
                      model,
                      model_output_log_prob,
                      model_lstm_state,
                      session,
                      direction):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB._score()")

    # convert symbols into id
    candidate_list = [[deepnovo_config.vocab[x] for x in candidate] 
                      for candidate in candidate_list]

    # we shall group candidates into minibatches
    # === candidate_len ===
    # s
    # i
    # z
    # e
    # =====================
    minibatch_size = len(candidate_list) # number of candidates
    candidate_len = len(candidate_list[0]) # length of each candidate

    # candidates share the same state0, so repeat into [minibatch_size, 512]
    # the states will also be updated after every iteration
    state0_c = state0_c.reshape((1, -1)) # reshape to [1, 512]
    state0_h = state0_h.reshape((1, -1))
    minibatch_state_c = np.repeat(state0_c, minibatch_size, axis=0)
    minibatch_state_h = np.repeat(state0_h, minibatch_size, axis=0)

    # mass of each candidate, will be accumulated everytime an AA is appended
    minibatch_prefix_mass = np.zeros(minibatch_size)

    # output is a list of candidate_len arrays of shape [minibatch_size, 26]
    # each row is log of probability distribution over 26 classes/symbols
    output_logprob_list = []

    # recurrent iterations
    for position in range(candidate_len):

      # gather minibatch data
      minibatch_AA_id = np.zeros(minibatch_size)
      for index, candidate in enumerate(candidate_list):
        AA = candidate[position]
        minibatch_AA_id[index] = AA
        minibatch_prefix_mass[index] += deepnovo_config.mass_ID[AA]
      # this is the most time-consuming ~70-75%
      minibatch_intensity = [get_candidate_intensity(spectrum_original,
                                                     precursor_mass,
                                                     prefix_mass,
                                                     direction)
                             for prefix_mass in np.nditer(minibatch_prefix_mass)]
      # final shape [minibatch_size, 26, 8, 10]
      minibatch_intensity = np.array(minibatch_intensity)

      # model feed
      input_feed = {}
      input_feed[model.input_AA_id[1].name] = minibatch_AA_id
      input_feed[model.input_intensity.name] = minibatch_intensity
      input_feed[model.input_state[0].name] = minibatch_state_c
      input_feed[model.input_state[1].name] = minibatch_state_h
      # and run
      output_feed = [model_output_log_prob, model_lstm_state]
      output_logprob, (minibatch_state_c, minibatch_state_h) = session.run(
          fetches=output_feed,
          feed_dict=input_feed)

      output_logprob_list.append(output_logprob)

    return output_logprob_list


  def _compute_peptide_mass(self, peptide):
    """TODO(nh2tran): docstring.
    """

    #~ print("".join(["="] * 80)) # section-separating line ===
    #~ print("WorkerDB._compute_peptide_mass()")

    peptide_mass = (deepnovo_config.mass_N_terminus
                    + sum(deepnovo_config.mass_AA[aa] for aa in peptide)
                    + deepnovo_config.mass_C_terminus)

    return peptide_mass


  def _expand_peptide_modification(self, peptide):
    """TODO(nh2tran): docstring.
       May also use parser.isoforms
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB._expand_peptide_modification()")

    # recursively add all modifications
    pepmod_list = [peptide] # the first entry without any modifications
    mod_count = 0
    for position, aa in enumerate(peptide):
      if aa in self.var_mod_list:
        mod_count += 1
        # add modification of this position to all peptides in the current list
        new_mod_list = []
        for pepmod in pepmod_list:
          new_mod = pepmod[:]
          new_mod[position] = aa + 'mod'
          new_mod_list.append(new_mod)
        pepmod_list = pepmod_list + new_mod_list
    # sanity check of the recursive iteration
    assert len(pepmod_list) == pow(2, mod_count), (
        "Wrong peptide expansion!")

    return pepmod_list


  def _filter_by_mass(self, precursor_mass):
    """TODO(nh2tran): docstring.
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB._filter_by_mass()")

    # use ppm instead of absolute mass_tolerance
    #~ mass_tolerance = self.mass_tolerance
    mass_tolerance = self.ppm * precursor_mass

    # 1st filter by the peptide mass and the max pepmod mass
    filter1_index = np.flatnonzero(np.logical_and(
        np.less_equal(self.peptide_mass_array,
                      precursor_mass + mass_tolerance),
        np.greater_equal(self.pepmod_maxmass_array,
                         precursor_mass - mass_tolerance)))

    # find all possible modifications
    pepmod_list = []
    for index in filter1_index:
      peptide = self.peptide_list[index]
      pepmod_list += self._expand_peptide_modification(peptide)
    pepmod_mass_array = np.array([self._compute_peptide_mass(pepmod)
                                  for pepmod in pepmod_list])

    # 2nd filter by exact pepmod mass
    filter2_index = np.flatnonzero(np.logical_and(
        np.less_equal(pepmod_mass_array,
                      precursor_mass + mass_tolerance),
        np.greater_equal(pepmod_mass_array,
                         precursor_mass - mass_tolerance)))

    candidate_list = [pepmod_list[x] for x in filter2_index]

    return candidate_list

