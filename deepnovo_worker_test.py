# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

import numpy as np

import deepnovo_config

class WorkerTest(object):
  """TODO(nh2tran): docstring.
     The WorkerTest should be stand-alone and separated from other workers.
  """


  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerTest.__init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.MZ_MAX = deepnovo_config.MZ_MAX

    self.target_file = deepnovo_config.target_file
    self.predicted_file = deepnovo_config.predicted_file
    self.predicted_format = deepnovo_config.predicted_format
    self.accuracy_file = deepnovo_config.accuracy_file
    print("input_file = {0:s}".format(self.target_file))
    print("predicted_file = {0:s}".format(self.predicted_file))
    print("predicted_format = {0:s}".format(self.predicted_format))
    print("accuracy_file = {0:s}".format(self.accuracy_file))

    self.target_dict = {}
    self.predicted_list = []


  def test_accuracy(self, db_peptide_list):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerTest.test_accuracy()")

    self._get_target()
    target_count_total = len(self.target_dict)
    target_len_total = sum([len(x) for x in self.target_dict.itervalues()])

    # this part is tricky!
    # some target peptides are reported by PEAKS DB but not found in
    #   db_peptide_list due to mistakes in cleavage rules.
    # we only consider target peptides in db_peptide_list for the moment.
    target_dict_db = {}
    for scan, target in self.target_dict.iteritems():
      target_simplied = target
      # remove the extension 'mod' from variable modifications
      target_simplied = ['M' if x=='Mmod' else x for x in target_simplied]
      target_simplied = ['N' if x=='Nmod' else x for x in target_simplied]
      target_simplied = ['Q' if x=='Qmod' else x for x in target_simplied]
      if target_simplied in db_peptide_list:
        target_dict_db[scan] = target
      else:
        print("target not found: ", target_simplied)
    target_count_db = len(target_dict_db)
    target_len_db = sum([len(x) for x in target_dict_db.itervalues()])

    # note that the prediction has already skipped precursor_mass > MZ_MAX
    self._get_predicted()
    predicted_count_mass = len(self.predicted_list)
    predicted_len_mass = sum([len(x["sequence"]) for x in self.predicted_list])

    # we skip target peptides with precursor_mass > MZ_MAX
    target_count_db_mass = 0
    target_len_db_mass = 0
    # we also skip predicted peptides whose scans are not in target_dict_db
    predicted_count_mass_db = 0
    predicted_len_mass_db = 0
    # the recall is calculated on remaining peptides
    recall_AA_total = 0.0
    recall_peptide_total = 0.0

    for index, predicted in enumerate(self.predicted_list):

      scan = predicted["scan"]
      if scan in target_dict_db:

        target = target_dict_db[scan]
        target_count_db_mass += 1
        target_len= len(target)
        target_len_db_mass += target_len

        predicted_count_mass_db += 1
        predicted_len= len(predicted["sequence"])
        predicted_len_mass_db += predicted_len
  
        predicted_AA_id = [deepnovo_config.vocab[x] for x in predicted["sequence"]]
        target_AA_id = [deepnovo_config.vocab[x] for x in target]
        recall_AA = self._match_AA_novor(target_AA_id, predicted_AA_id)
        recall_AA_total += recall_AA
        if recall_AA == target_len:
          recall_peptide_total += 1
        else:
          print("index = ", index)
          print(scan)
          print(target)
          print(predicted["sequence"], predicted["score"])
          print(recall_AA)

    print("target_count_total = {0:d}".format(target_count_total))
    print("target_len_total = {0:d}".format(target_len_total))
    print("target_count_db = {0:d}".format(target_count_db))
    print("target_len_db = {0:d}".format(target_len_db))
    print("target_count_db_mass: {0:d}".format(target_count_db_mass))
    print("target_len_db_mass: {0:d}".format(target_len_db_mass))
    print()

    print("predicted_count_mass: {0:d}".format(predicted_count_mass))
    print("predicted_len_mass: {0:d}".format(predicted_len_mass))
    print("predicted_count_mass_db: {0:d}".format(predicted_count_mass_db))
    print("predicted_len_mass_db: {0:d}".format(predicted_len_mass_db))
    print()

    print("recall_AA_total = {0:.4f}".format(recall_AA_total / target_len_total))
    print("recall_AA_db = {0:.4f}".format(recall_AA_total / target_len_db))
    print("recall_AA_db_mass = {0:.4f}".format(recall_AA_total / target_len_db_mass))
    print("recall_peptide_total = {0:.4f}".format(recall_peptide_total / target_count_total))
    print("recall_peptide_db = {0:.4f}".format(recall_peptide_total / target_count_db))
    print("recall_peptide_db_mass = {0:.4f}".format(recall_peptide_total / target_count_db_mass))
    print("precision_AA_mass_db  = {0:.4f}".format(recall_AA_total / predicted_len_mass_db))
  
  
  def _get_predicted(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerTest._get_predicted()")

    predicted_list = []
    with open(self.predicted_file, 'r') as handle:
      # header
      handle.readline()
      for line in handle:
        line_split = re.split('\t|\n', line)
        predicted = {}
        predicted["scan"] = line_split[0]
        predicted["sequence"] = re.split(',', line_split[1])
        predicted["score"] = float(line_split[2])
        predicted["position_score"] = [float(x)
                                       for x in re.split(',', line_split[3])]
        predicted_list.append(predicted)

    self.predicted_list = predicted_list


  def _get_target(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerTest._get_target()")

    target_dict = {}
    with open(self.target_file, 'r') as handle:
      for line in handle:
        if "SCANS=" in line:
          scan = re.split('=|\n', line)[1]
        elif "SEQ=" in line:
          raw_sequence = re.split('=|\n', line)[1]
          peptide = self._parse_sequence(raw_sequence)
          target_dict[scan] = peptide
        else:
          print("Error: wrong target format.")
          sys.exit()

    self.target_dict = target_dict


  def _parse_sequence(self, raw_sequence):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerTest._parse_sequence()")

    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
      if raw_sequence[index] == "(":
        if peptide[-1] == "C" and raw_sequence[index:index+8] == "(+57.02)":
          peptide[-1] = "Cmod"
          index += 8
        elif peptide[-1] == 'M' and raw_sequence[index:index+8] == "(+15.99)":
          peptide[-1] = 'Mmod'
          index += 8
        elif peptide[-1] == 'N' and raw_sequence[index:index+6] == "(+.98)":
          peptide[-1] = 'Nmod'
          index += 6
        elif peptide[-1] == 'Q' and raw_sequence[index:index+6] == "(+.98)":
          peptide[-1] = 'Qmod'
          index += 6
        else: # unknown modification
          print("ERROR: unknown modification!")
          print("raw_sequence = ", raw_sequence)
          sys.exit()
      else:
        peptide.append(raw_sequence[index])
        index += 1

    return peptide


  def _match_AA_novor(self, target, predicted):
    """TODO(nh2tran): docstring."""
  
    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerTest._test_AA_match_novor()")

    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [deepnovo_config.mass_ID[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [deepnovo_config.mass_ID[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)
  
    i = 0
    j = 0
    while i < target_len and j < predicted_len:
      if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
        if abs(target_mass[i] - predicted_mass[j]) < 0.1:
        #~ if  decoder_input[index_aa] == output[index_aa]:
          num_match += 1
        i += 1
        j += 1
      elif target_mass_cum[i] < predicted_mass_cum[j]:
        i += 1
      else:
        j += 1

    return num_match
  
