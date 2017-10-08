# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import re

import numpy as np
import deepnovo_config

random.seed(0)
np.random.seed(0)


def cat_file_mgf(input_file_list, fraction_list, output_file):
  """TODO(nh2tran): docstring."""

  print("cat_file_mgf()")

  counter = 0

  with open(output_file, mode="w") as output_handle:

    for index, input_file in enumerate(input_file_list):

      print("input_file = ", os.path.join(input_file))

      with open(input_file, mode="r") as input_handle:

        line = input_handle.readline()
        while line:

          if "SCANS=" in line: # a spectrum found
            counter += 1
            scan = int(re.split('=', line)[1])
            # re-number scan id
            output_handle.write("SCANS=F{0}:{1}\n".format(
                fraction_list[index], scan))

          else:
            output_handle.write(line)

          line = input_handle.readline()

  print("counter ", counter)


#~ number_fraction = 72
#~ cat_file_mgf(["data.training/yeast.low.takeda_2015/peaks.db/"
              #~ + str(i) + "_frac.mgf"
              #~ for i in range(1, number_fraction + 1)],
             #~ range(1, number_fraction + 1),
             #~ "data.training/yeast.low.takeda_2015/peaks.db.mgf")


def partition_train_valid_test_dup_mgf(input_file, prob):
  """TODO(nh2tran): docstring.
     Partition a dataset into three random sets train-valid-test with a
     distribution, e.g. 90-5-5 percent.
  """

  print("partition_train_valid_test_dup_mgf()")
  print("input_file = ", os.path.join(input_file))
  print("prob = ", prob)

  output_file_train = input_file + ".train" + ".dup"
  output_file_valid = input_file + ".valid" + ".dup"
  output_file_test = input_file + ".test" + ".dup"

  with open(input_file, mode="r") as input_handle:
    with open(output_file_train, mode="w") as output_handle_train:
      with open(output_file_valid, mode="w") as output_handle_valid:
        with open(output_file_test, mode="w") as output_handle_test:

          counter = 0
          counter_train = 0
          counter_valid = 0
          counter_test = 0

          line = input_handle.readline()
          while line:

            if "BEGIN IONS" in line: # a spectrum found

              counter += 1

              set_num = np.random.choice(a=3, size=1, p=prob)
              if set_num == 0:
                output_handle = output_handle_train
                counter_train += 1
              elif set_num == 1:
                output_handle = output_handle_valid
                counter_valid += 1
              else:
                output_handle = output_handle_test
                counter_test += 1

            output_handle.write(line)
            line = input_handle.readline()

  input_handle.close()
  output_handle_train.close()
  output_handle_valid.close()
  output_handle_test.close()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)


#~ partition_train_valid_test_dup_mgf(
    #~ "data.training/yeast.low.coon_2013/peaks.db.mgf.test.dup",
    #~ [0.10, 0.90, 0.00])


def partition_train_valid_test_unique_mgf(input_file, prob):
  """TODO(nh2tran): docstring.
     Partition a dataset into three random sets train-valid-test with a
     distribution, e.g. 90-5-5 percent.
     This version removes all duplicated peptides so that each peptide has only
     one spectrum (selected randomly).
  """

  print("partition_train_valid_test_unique_mgf()")
  print("input_file = ", os.path.join(input_file))
  print("prob = ", prob)

  output_file_train = input_file + ".train" + ".unique"
  output_file_valid = input_file + ".valid" + ".unique"
  output_file_test = input_file + ".test" + ".unique"

  peptide_list = []

  with open(input_file, mode="r") as input_handle:
    with open(output_file_train, mode="w") as output_handle_train:
      with open(output_file_valid, mode="w") as output_handle_valid:
        with open(output_file_test, mode="w") as output_handle_test:

          counter = 0
          counter_train = 0
          counter_valid = 0
          counter_test = 0

          line = input_handle.readline()
          while line:

            if "BEGIN IONS" in line: # a spectrum found

              line_buffer = []
              line_buffer.append(line)
              # TITLE
              line = input_handle.readline()
              line_buffer.append(line)
              # PEPMASS
              line = input_handle.readline()
              line_buffer.append(line)
              # CHARGE
              line = input_handle.readline()
              line_buffer.append(line)
              # SCANS
              line = input_handle.readline()
              line_buffer.append(line)
              # RTINSECONDS
              line = input_handle.readline()
              line_buffer.append(line)
              # SEQ
              line = input_handle.readline()
              line_buffer.append(line)
              peptide = re.split('=|\n|\r', line)[1]

              if not peptide in peptide_list: # new peptide

                peptide_list.append(peptide)
                counter += 1

                set_num = np.random.choice(a=3, size=1, p=prob)
                if set_num == 0:
                  output_handle = output_handle_train
                  counter_train += 1
                elif set_num == 1:
                  output_handle = output_handle_valid
                  counter_valid += 1
                else:
                  output_handle = output_handle_test
                  counter_test += 1

                for l in line_buffer:
                  output_handle.write(l)

                while line and not "END IONS" in line:
                  line = input_handle.readline()
                  output_handle.write(line)

                output_handle.write("\n")

            line = input_handle.readline()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)


#~ partition_train_valid_test_unique_mgf("data/human.PXD002179.sds/peaks.db.mgf",
                                      #~ [1.0, 0.0, 0.0])


def partition_train_valid_test_unique_control_mgf(input_file,
                                                  prob,
                                                  max_spectra_per_peptide):
  """TODO(nh2tran): docstring.
     Partition a dataset into three random sets train-valid-test with a
     distribution, e.g. 90-5-5 percent.
     This version removes duplicated peptides so that each peptide has at most
     max_spectra_per_peptide (selected randomly).
  """

  print("partition_train_valid_test_unique_control_mgf()")
  print("input_file = ", os.path.join(input_file))
  print("prob = ", prob)

  output_file_train = (input_file + ".train" + ".unique"
                       + str(max_spectra_per_peptide))
  output_file_valid = (input_file + ".valid" + ".unique"
                       + str(max_spectra_per_peptide))
  output_file_test = (input_file + ".test" + ".unique"
                      + str(max_spectra_per_peptide))

  peptide_list = []
  peptide_spectra_count = {}

  with open(input_file, mode="r") as input_handle:
    with open(output_file_train, mode="w") as output_handle_train:
      with open(output_file_valid, mode="w") as output_handle_valid:
        with open(output_file_test, mode="w") as output_handle_test:

          counter = 0
          counter_train = 0
          counter_valid = 0
          counter_test = 0

          line = input_handle.readline()
          while line:

            if "BEGIN IONS" in line: # a spectrum found

              line_buffer = []
              line_buffer.append(line)
              # TITLE
              line = input_handle.readline()
              line_buffer.append(line)
              # PEPMASS
              line = input_handle.readline()
              line_buffer.append(line)
              # CHARGE
              line = input_handle.readline()
              line_buffer.append(line)
              # SCANS
              line = input_handle.readline()
              line_buffer.append(line)
              # RTINSECONDS
              line = input_handle.readline()
              line_buffer.append(line)
              # SEQ
              line = input_handle.readline()
              line_buffer.append(line)
              peptide = re.split('=|\n|\r', line)[1]

              if not peptide in peptide_list: # new peptide
                peptide_list.append(peptide)
                peptide_spectra_count[peptide] = 0

              if peptide_spectra_count[peptide] < max_spectra_per_peptide:

                peptide_spectra_count[peptide] += 1
                counter += 1

                set_num = np.random.choice(a=3, size=1, p=prob)
                if set_num == 0:
                  output_handle = output_handle_train
                  counter_train += 1
                elif set_num == 1:
                  output_handle = output_handle_valid
                  counter_valid += 1
                else:
                  output_handle = output_handle_test
                  counter_test += 1

                for l in line_buffer:
                  output_handle.write(l)

                while line and not "END IONS" in line:
                  line = input_handle.readline()
                  output_handle.write(line)

                output_handle.write("\n")

            line = input_handle.readline()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)


#~ partition_train_valid_test_unique_control_mgf(
    #~ "data/human.cancer/peaks.db.frac_1_10.mgf",
    #~ [1.0, 0.0, 0.0],
    #~ 4)


def partition_train_valid_test_repeat_mgf(input_file, prob):
  """TODO(nh2tran): docstring.
     Partition a dataset into three random sets train-valid-test with a
     distribution, e.g. 90-5-5 percent.
     Each peptide may correspond to multiple different spectra but the three
     sets do not share common peptides.
  """

  print("partition_train_valid_test_repeat_mgf()")
  print("input_file = ", os.path.join(input_file))
  print("prob = ", prob)

  output_file_train = input_file + ".train" + ".repeat"
  output_file_valid = input_file + ".valid" + ".repeat"
  output_file_test = input_file + ".test" + ".repeat"

  peptide_train_list = []
  peptide_valid_list = []
  peptide_test_list = []

  with open(input_file, mode="r") as input_handle:
    with open(output_file_train, mode="w") as output_handle_train:
      with open(output_file_valid, mode="w") as output_handle_valid:
        with open(output_file_test, mode="w") as output_handle_test:

          counter = 0
          counter_train = 0
          counter_valid = 0
          counter_test = 0
          counter_unique = 0

          line = input_handle.readline()
          while line:

            if "BEGIN IONS" in line: # a spectrum found

              line_buffer = []
              line_buffer.append(line)
              # TITLE
              line = input_handle.readline()
              line_buffer.append(line)
              # PEPMASS
              line = input_handle.readline()
              line_buffer.append(line)
              # CHARGE
              line = input_handle.readline()
              line_buffer.append(line)
              # SCANS
              line = input_handle.readline()
              line_buffer.append(line)
              # RTINSECONDS
              line = input_handle.readline()
              line_buffer.append(line)
              # SEQ
              line = input_handle.readline()
              line_buffer.append(line)
              peptide = re.split('=|\n|\r', line)[1]

              # found a spectrum and a peptide
              counter += 1

              # check if the peptide already exists in any of the three lists
              # if yes, this new spectrum will be assigned to that list
              if peptide in peptide_train_list:
                output_handle = output_handle_train
                counter_train += 1
              elif peptide in peptide_valid_list:
                output_handle = output_handle_valid
                counter_valid += 1
              elif peptide in peptide_test_list:
                output_handle = output_handle_test
                counter_test += 1
              # if not, this new peptide and its spectrum will be randomly
              # assigned
              else:
                counter_unique += 1
                set_num = np.random.choice(a=3, size=1, p=prob)
                if set_num == 0:
                  peptide_train_list.append(peptide)
                  output_handle = output_handle_train
                  counter_train += 1
                elif set_num == 1:
                  peptide_valid_list.append(peptide)
                  output_handle = output_handle_valid
                  counter_valid += 1
                else:
                  peptide_test_list.append(peptide)
                  output_handle = output_handle_test
                  counter_test += 1

              for l in line_buffer:
                output_handle.write(l)

              while line and not "END IONS" in line:
                line = input_handle.readline()
                output_handle.write(line)

              output_handle.write("\n")

            line = input_handle.readline()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)
  print("counter_unique ", counter_unique)


#~ partition_train_valid_test_repeat_mgf(
    #~ "data.training/yeast.low.heinemann_2015/peaks.db.mgf",
    #~ [0.90, 0.05, 0.05])


def prepare_test_file(input_file):
  """TODO(nh2tran): docstring.
     Filter spectra with MZ_MAX, unknown_modification.
     Extract ground-truth peptide sequences from database-search.
  """

  print("prepare_test_file()")
  print("input_file = ", os.path.join(input_file))

  dbseq_file = input_file + ".dbseq"
  print("dbseq_file = ", dbseq_file)

  counter = 0
  counter_skipped = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_skipped_mass = 0

  with open(input_file, mode="r") as input_handle:
    with open(dbseq_file, mode="w") as dbseq_handle:

      print("scan \t target_seq \n", file=dbseq_handle, end="")

      line = input_handle.readline()
      while line:

        if "BEGIN IONS" in line: # a spectrum found

          line_buffer = []
          line_buffer.append(line)

          unknown_modification = False

          # header TITLE
          line = input_handle.readline()
          line_buffer.append(line)

          # header PEPMASS
          line = input_handle.readline()
          peptide_ion_mz = float(re.split('=|\n', line)[1])
          line_buffer.append(line)

          # header CHARGE
          line = input_handle.readline()
          charge = float(re.split('=|\+', line)[1]) # pylint: disable=anomalous-backslash-in-string
          line_buffer.append(line)

          # header SCANS
          line = input_handle.readline()
          #~ scan = int(re.split('=', line)[1])
          scan = re.split('=|\n', line)[1]
          line_buffer.append(line)

          # header RTINSECONDS
          line = input_handle.readline()
          line_buffer.append(line)

          # header SEQ
          line = input_handle.readline()
          line_buffer.append(line)
          raw_sequence = re.split('=|\n|\r', line)[1]
          raw_sequence_len = len(raw_sequence)
          peptide = []
          index = 0
          while index < raw_sequence_len:
            if raw_sequence[index] == "(":
              if (peptide[-1] == "C"
                  and raw_sequence[index:index+8] == "(+57.02)"):
                peptide[-1] = "Cmod"
                index += 8
              elif (peptide[-1] == 'M'
                    and raw_sequence[index:index+8] == "(+15.99)"):
                peptide[-1] = 'Mmod'
                index += 8
              elif (peptide[-1] == 'N'
                    and raw_sequence[index:index+6] == "(+.98)"):
                peptide[-1] = 'Nmod'
                index += 6
              elif (peptide[-1] == 'Q'
                    and raw_sequence[index:index+6] == "(+.98)"):
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

          # skip if unknown_modification
          if unknown_modification:
            counter_skipped += 1
            counter_skipped_mod += 1
            continue

          # skip if neutral peptide_mass > MZ_MAX(3000.0)
          peptide_mass = peptide_ion_mz*charge - charge*deepnovo_config.mass_H
          if peptide_mass > deepnovo_config.MZ_MAX:
            counter_skipped += 1
            counter_skipped_mass += 1
            continue

          # TRAINING-SKIP: skip if peptide length > MAX_LEN(30)
          # TESTING-ERROR: not allow peptide length > MAX_LEN(50)
          peptide_len = len(peptide)
          if peptide_len > deepnovo_config.MAX_LEN:
            print("ERROR: peptide_len {0} exceeds MAX_LEN {1}".format(
                peptide_len,
                deepnovo_config.MAX_LEN))
            sys.exit()
            #~ counter_skipped += 1
            #~ counter_skipped_len += 1
            #~ continue

          # AN ENTRY FOUND
          counter += 1
          if counter % 10000 == 0:
            print("  reading peptide %d" % counter)

          # output ground-truth peptide sequence
          print("%s\t%s\n" % (scan, ",".join(peptide)),
                file=dbseq_handle,
                end="")

          while line and not "END IONS" in line:
            line = input_handle.readline()

        line = input_handle.readline()

  print("  total peptide read %d" % counter)
  print("  total peptide skipped %d" % counter_skipped)
  print("  total peptide skipped by mod %d" % counter_skipped_mod)
  print("  total peptide skipped by len %d" % counter_skipped_len)
  print("  total peptide skipped by mass %d" % counter_skipped_mass)


#~ prepare_test_file("data.training/yeast.low.takeda_2015/peaks.db.mgf")


def partition_dbseq(dbseq_file, trainseq_file):
  """TODO(nh2tran): docstring.
     Partition a dbseq file into 2 sets: overlapping & nonoverlapping with the
     trainseq file.
  """

  print("partition_dbseq()")
  print("dbseq_file = ", dbseq_file)
  print("trainseq_file = ", trainseq_file)

  trainseq = []

  with open(trainseq_file, mode="r") as trainseq_handle:

    # header
    trainseq_handle.readline()
    for line in trainseq_handle:
      line_split = re.split('\t|\n', line)
      #~ scan = line_split[0]
      peptide = line_split[1]
      trainseq.append(peptide)

  overlap_file = dbseq_file + ".overlap"
  nonoverlap_file = dbseq_file + ".nonoverlap"
  count = 0
  count_overlap = 0
  count_nonoverlap = 0

  with open(dbseq_file, mode="r") as dbseq_handle:
    with open(overlap_file, mode="w") as overlap_handle:
      with open(nonoverlap_file, mode="w") as nonoverlap_handle:

        # header
        line = dbseq_handle.readline()
        overlap_handle.write(line)
        nonoverlap_handle.write(line)

        for line in dbseq_handle:

          line_split = re.split('\t|\n', line)
          #~ scan = line_split[0]
          peptide = line_split[1]

          if peptide in trainseq:
            overlap_handle.write(line)
            count_overlap += 1
          else:
            nonoverlap_handle.write(line)
            count_nonoverlap += 1

          count += 1

  print("count = {0:d}".format(count))
  print("count_overlap = {0:d}".format(count_overlap))
  print("count_nonoverlap = {0:d}".format(count_nonoverlap))


#~ partition_dbseq("data/human.cancer/peaks.db.frac_21_41.mgf.dbseq",
                #~ "data/human.cancer/peaks.db.frac_1_20.mgf.dbseq")


def read_dbseq(dbseq_file):
  """TODO(nh2tran): docstring."""

  print("read_dbseq()")
  print("dbseq_file = ", dbseq_file)

  dbseq = {}
  batch_len_AA = 0.0

  with open(dbseq_file, mode="r") as dbseq_handle:

    # header
    dbseq_handle.readline()

    for line in dbseq_handle:
      line_split = re.split('\t|\n', line)
      scan = line_split[0]
      peptide = re.split(',', line_split[1])
      dbseq[scan] = [deepnovo_config.vocab[x] for x in peptide]
      batch_len_AA += len(peptide)

  batch_size = len(dbseq)
  print("batch_size = ", batch_size)
  print("batch_len_AA = ", batch_len_AA)

  return dbseq, batch_size, batch_len_AA


def read_novonet(novonet_file):
  """TODO(nh2tran): docstring."""

  print("read_novonet()")
  print("novonet_file = ", novonet_file)

  novonet = {}

  with open(novonet_file, mode="r") as novonet_handle:

    # header
    novonet_handle.readline()

    for line in novonet_handle:
      line_split = re.split('\t|\n', line)
      scan = line_split[0]
      if line_split[2] == "": # empty output
        novonet_seq_id = []
      else:
        novonet_seq = re.split(',', line_split[2])
        novonet_seq_id = [deepnovo_config.vocab[x] for x in novonet_seq]
      novonet[scan] = novonet_seq_id

  return novonet


def read_peaks(peaks_denovo_file, peaks_format, alc_threshold):
  """TODO(nh2tran): docstring."""

  print("read_peaks()")
  print("peaks_denovo_file = ", peaks_denovo_file)

  if peaks_format == "old_7.5":
    peptide_column = 1
    alc_column = 3
  elif peaks_format == "new_8.0":
    peptide_column = 3
    alc_column = 5
  else:
    print("ERROR: wrong PEAKS denovo format")
    sys.exit()

  peaks = {}
  peaks_raw = {}

  with open(peaks_denovo_file, mode="r") as peaks_handle:

    # header
    peaks_handle.readline()

    for line in peaks_handle:

      line_split = re.split(",", line)

      if peaks_format == "old_7.5":
        scan = line_split[0]
      elif peaks_format == "new_8.0":
        scan = "F" + line_split[0] + ":" + line_split[1]

      if line_split[peptide_column] == "": # empty output
        peaks_seq_id = []
      else:
        raw_sequence = line_split[peptide_column]
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
            #~ elif ("".join(raw_sequence[index:index+8])=="(+42.01)"):
              #~ print("ERROR: unknown modification!")
              #~ print("raw_sequence = ", raw_sequence)
              #~ sys.exit()
              unknown_modification = True
              break
          else:
            peptide.append(raw_sequence[index])
            index += 1

        peaks_seq_id = [deepnovo_config.vocab[x] for x in peptide]

      alc_score = float(line_split[alc_column])
      if alc_score >= alc_threshold:
        peaks[scan] = peaks_seq_id
        peaks_raw[scan] = raw_sequence

  return peaks, peaks_raw


def get_peaks_denovo_spectra(output_spectra_file,
                             raw_spectra_file,
                             peaks_denovo_file,
                             peaks_format,
                             alc_threshold=0):
  """TODO(nh2tran): docstring."""

  print("get_peaks_denovo_spectra()")
  print("peaks_denovo_file = ", peaks_denovo_file)
  print("ALC cut-off = ", alc_threshold)
  print("raw_spectra_file = ", raw_spectra_file)

  _, peaks_denovo_peptides = read_peaks(peaks_denovo_file,
                                        peaks_format,
                                        alc_threshold)
  print("peaks_denovo_peptides: ", len(peaks_denovo_peptides))

  counter_spectra = 0

  with open(raw_spectra_file, mode="r") as input_handle:
    with open(output_spectra_file, mode="w") as output_handle:

      line = input_handle.readline()
      while line:

        if "BEGIN IONS" in line: # a spectrum found

          line_buffer = []
          line_buffer.append(line)
          # header TITLE
          line = input_handle.readline()
          line_buffer.append(line)
          # header PEPMASS
          line = input_handle.readline()
          line_buffer.append(line)
          # header CHARGE
          line = input_handle.readline()
          line_buffer.append(line)
          # header SCANS
          line = input_handle.readline()
          #~ scan = int(re.split('=', line)[1])
          scan = re.split('=|\n', line)[1]
          line_buffer.append(line)

          # lookup scan id
          if not scan in peaks_denovo_peptides:
            continue
          else:

            counter_spectra += 1

            for l in line_buffer:
              output_handle.write(l)

            # RTINSECONDS
            line = input_handle.readline()
            output_handle.write(line)

            # SEQ
            line = "SEQ=" + peaks_denovo_peptides[scan] + "\n"
            output_handle.write(line)

            while line and not "END IONS" in line:
              line = input_handle.readline()
              output_handle.write(line)

            output_handle.write("\n")

        line = input_handle.readline()

  print("total spectra found %d" % counter_spectra)


def test_AA_match_novor(decoder_input, output):
  """TODO(nh2tran): docstring."""

  num_match = 0
  decoder_input_len = len(decoder_input)
  output_len = len(output)
  decoder_input_mass = [deepnovo_config.mass_ID[x] for x in decoder_input]
  decoder_input_mass_cum = np.cumsum(decoder_input_mass)
  output_mass = [deepnovo_config.mass_ID[x] for x in output]
  output_mass_cum = np.cumsum(output_mass)

  i = 0
  j = 0
  while i < decoder_input_len and j < output_len:
    if abs(decoder_input_mass_cum[i] - output_mass_cum[j]) < 0.5:
      if abs(decoder_input_mass[i] - output_mass[j]) < 0.1:
      #~ if  decoder_input[index_aa] == output[index_aa]:
        num_match += 1
      i += 1
      j += 1
    elif decoder_input_mass_cum[i] < output_mass_cum[j]:
      i += 1
    else:
      j += 1

  return num_match


def test_accuracy(dbseq_file,
                  denovo_file,
                  tool,
                  peaks_format=None,
                  alc_threshold=None):
  """TODO(nh2tran): docstring."""

  print("test_accuracy()")

  batch_accuracy_AA = 0.0
  batch_len_decode = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0

  dbseq, batch_size, batch_len_AA = read_dbseq(dbseq_file)

  if tool == "novonet":
    denovo = read_novonet(denovo_file)
  elif tool == "peaks":
    denovo, _ = read_peaks(denovo_file, peaks_format, alc_threshold)

  count_skipped = 0

  # for testing
  test_output = dict.fromkeys(dbseq.keys(), [])

  for scan, seq in denovo.iteritems():

    if scan in dbseq:

      accuracy_AA = test_AA_match_novor(dbseq[scan], seq)
      len_AA = len(dbseq[scan])
      # for testing
      output_seq = [deepnovo_config.vocab_reverse[x] for x in seq]
      test_output[scan] = [output_seq, accuracy_AA]

      len_decode = len(seq)
      batch_len_decode += len_decode
      batch_accuracy_AA += accuracy_AA
      #~ batch_accuracy_AA += accuracy_AA/len_AA
      if accuracy_AA == len_AA:
        num_exact_match += 1.0
      if len(seq) == len_AA:
        num_len_match += 1.0

    else:
      count_skipped += 1

  # for testing
  with open("test_accuracy.tab", "w") as file_handle:

    file_handle.write("scan \t target_seq \t target_len \t output_seq \t "
                      "accuracy_AA \n")

    for scan, output in test_output.iteritems():
      target_seq = [deepnovo_config.vocab_reverse[x] for x in dbseq[scan]]
      target_len = len(target_seq)
      if not output:
        file_handle.write("{0:s}\t{1:s}\t{2:d}\t{3:s}\t{4:d}\n".format(
            scan,
            target_seq,
            target_len,
            [],
            0))
      else:
        file_handle.write("{0:s}\t{1:s}\t{2:d}\t{3:s}\t{4:d}\n".format(
            scan,
            target_seq,
            target_len,
            output[0],
            output[1]))

  print("  recall_AA %.4f" % (batch_accuracy_AA / batch_len_AA))
  #~ print("  accuracy_AA %.4f" % (batch_accuracy_AA / batch_size))
  print("  precision_AA %.4f" % (batch_accuracy_AA / batch_len_decode))
  print("  recall_peptide %.4f" % (num_exact_match / batch_size))
  print("  recall_len %.4f" % (num_len_match / batch_size))
  print("  count_skipped (not in dbseq) %d" % (count_skipped))


# NovoNet
#~ test_accuracy(
    #~ "data/yeast.full/peaks.db.frac_456.mgf.dbseq",
    #~ "train/train.intensity_only.yeast.full.db.frac_123.repeat/decode_output.db.tab",
    #~ "novonet")

# PEAKS
#~ test_accuracy(
    #~ "data.training/yeast.low.takeda_2015/peaks.db.mgf.dbseq",
    #~ "data.training/yeast.low.takeda_2015/peaks.denovo.csv",
    #~ "peaks",
    #peaks_format="old_7.5",
    #~ peaks_format="new_8.0",
    #~ alc_threshold=0)
