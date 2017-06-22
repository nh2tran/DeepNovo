from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import re

import numpy as np

#~ from tensorflow.models.rnn.translate import data_utils
import data_utils

random.seed(0)
np.random.seed(0)


def cat_file_mgf(input_file_list, fraction_list, output_file):
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
              output_handle.write("SCANS=F{0}:{1}\n".format(fraction_list[index], scan))
        
            else:

                output_handle.write(line)

            line = input_handle.readline()

  print("counter ", counter)

#~ cat_file_mgf(input_file_list=["data/human.data2/raw_1.mgf",
                              #~ "data/human.data2/raw_2.mgf",
                              #~ "data/human.data2/raw_3.mgf",
                              #~ "data/human.data2/raw_4.mgf",
                              #~ "data/human.data2/raw_5.mgf",
                              #~ "data/human.data2/raw_6.mgf",],
              #~ fraction_list=[1, 2, 3, 4, 5, 6],
              #~ output_file="data/human.data2/raw_spectra.mgf")
              
#~ frac=6  #record how many fractions need to be processed
#~ cat_file_mgf(input_file_list=["data/human.PXD002179.sds/" + str(i) + "_frac.txt" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/PXD004120_drosophilamelanogaster/peaks.db.frac_1_"+str(frac)+".mgf")
#~ cat_file_mgf(input_file_list=["data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine/" + str(i) + "_frac.mgf" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine.mgf")
#~ cat_file_mgf(input_file_list=["data/ab.testing/assem.public.mouse.light/peaks.refine/" + str(i) + "_frac.mgf" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/ab.testing/assem.public.mouse.light/peaks.refine.mgf")
#~ cat_file_mgf(input_file_list=["data/ab.testing/assem.public.human.light/peaks.refine/" + str(i) + "_frac.mgf" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/ab.testing/assem.public.human.light/peaks.refine.mgf")

#~ cat_file_mgf(input_file_list=["data/ab.testing/assem.public.human.heavy/peaks.refine/" + str(i) + "_frac.mgf" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/ab.testing/assem.public.human.heavy/peaks.refine.mgf")

#~ cat_file_mgf(input_file_list=["data/ab.testing/assem.public.mouse.heavy/peaks.refine/" + str(i) + "_frac.mgf" for i in range(1,frac+1)],
              #~ fraction_list=range(1,frac+1),
              #~ output_file="data/ab.testing/assem.public.mouse.heavy/peaks.refine.mgf")






# Partition a dataset into three random sets train-valid-test
#     with a distribution, e.g. 90-5-5 percent.
#
def partition_train_valid_test_dup_mgf(input_file, prob):
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

#~ partition_train_valid_test_dup_mgf(input_file="data/PXD004120_drosophilamelanogaster/peaks.db.frac_1_"+str(frac)+".mgf",
                                   #~ prob=[0.9, 0.05, 0.05])







# Partition a dataset into three random sets train-valid-test
#     with a distribution, e.g. 90-5-5 percent.
#
# This version removes all duplicated peptides so that 
#     each peptide has only one spectrum (selected randomly).
#
def partition_train_valid_test_unique_mgf(input_file, prob):
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
              
              if not (peptide in peptide_list): # new peptide

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

                while line and not ("END IONS" in line):
                  line = input_handle.readline()
                  output_handle.write(line)

                output_handle.write("\n")
                  
            line = input_handle.readline()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)






# Partition a dataset into three random sets train-valid-test
#     with a distribution, e.g. 90-5-5 percent
#
# Each peptide may correspond to multiple different spectra,
#     but the three sets do not share common peptides.
#
def partition_train_valid_test_repeat_mgf(input_file,prob):
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
              if (peptide in peptide_train_list):
                output_handle = output_handle_train
                counter_train += 1
              elif (peptide in peptide_valid_list):
                output_handle = output_handle_valid
                counter_valid += 1
              elif (peptide in peptide_test_list):
                output_handle = output_handle_test
                counter_test += 1
              # if not, this new peptide and its spectrum will be randomly assigned
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

              while line and not ("END IONS" in line):
                line = input_handle.readline()
                output_handle.write(line)

              output_handle.write("\n")
                  
            line = input_handle.readline()

  print("counter ", counter)
  print("counter_train ", counter_train)
  print("counter_valid ", counter_valid)
  print("counter_test ", counter_test)
  print("counter_unique ", counter_unique)

#~ partition_train_valid_test_repeat_mgf(input_file="data/human.cancer/peaks.db.frac_123456.mgf",
                                   #~ prob=[0.9, 0.05, 0.05])







# filter spectra with 
#     MAX_LEN = 20 aa
#     MZ_MAX = 3000 Da 
#     PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01 Da
#
# re-number scan id to unique because the input file was merged from 
#     different fractions
#
#
# and extract ground-truth peptide sequences from database-search
#
def prepare_test_file(input_file):
  print("prepare_test_file()")
  
  print("input_file = ", os.path.join(input_file))
  #
  #~ output_file = input_file + ".filter"
  #~ print("output_file = ", output_file)
  #
  dbseq_file = input_file + ".dbseq"
  print("dbseq_file = ", dbseq_file)

  counter = 0
  counter_skipped = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_skipped_mass = 0
  
  with open(input_file, mode="r") as input_handle:
    #~ with open(output_file, mode="w") as output_handle:
      with open(dbseq_file, mode="w") as dbseq_handle:
        
        print("scan \t target_seq \n", file=dbseq_handle, end="")

        line = input_handle.readline()
        while line:
  
          if "BEGIN IONS" in line: # a spectrum found
  
            line_buffer = []
            line_buffer.append(line)
        
            unknown_modification = False
            max_intensity = 0.0
  
            # header TITLE
            line = input_handle.readline()
            line_buffer.append(line)
  
            # header PEPMASS
            line = input_handle.readline()
            peptide_ion_mz = float(re.split('=|\n', line)[1])
            line_buffer.append(line)
  
            # header CHARGE
            line = input_handle.readline()
            charge = float(re.split('=|\+', line)[1])
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
            #~ peptide = peptide.translate(None, '+-.0123456789)') # modification signal "("
            #~ peptide = peptide.translate(None, '(+-.0123456789)') # ignore modifications
            raw_sequence_len = len(raw_sequence)
            peptide = []
            index = 0
            while (index < raw_sequence_len):
              if (raw_sequence[index]=="("):
                if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+57.02)"):
                #~ if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+58.01)"):
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
              print("ERROR: peptide_len {0} exceeds MAX_LEN {1}".format(peptide_len, data_utils.MAX_LEN))
              sys.exit()
              #~ counter_skipped += 1
              #~ counter_skipped_len += 1
              #~ continue
      
            # TRAINING-ONLY: testing peptide_mass & sequence_mass
            #~ sequence_mass = sum(data_utils.mass_AA[aa] for aa in peptide)
            #~ sequence_mass += data_utils.mass_N_terminus + data_utils.mass_C_terminus
            #~ if (abs(peptide_mass-sequence_mass) > data_utils.PRECURSOR_MASS_PRECISION_INPUT_FILTER):
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
              #~ counter_skipped += 1
              #~ continue
  
            # AN ENTRY FOUND
            counter += 1
            if counter % 10000 == 0:
              print("  reading peptide %d" % counter)
              
            # output ground-truth peptide sequence & re-number scan id
            print("%s\t%s\n" % (scan, ",".join(peptide)),
                  file=dbseq_handle,
                  end="")
      
            # output this data entry
            #~ for l in line_buffer:
              #~ output_handle.write(l)
            #
            while line and not ("END IONS" in line):
              line = input_handle.readline()
              #~ output_handle.write(line)
            #
            #~ output_handle.write("\n")
                
          line = input_handle.readline()
  
  print("  total peptide read %d" % counter)
  print("  total peptide skipped %d" % counter_skipped)
  print("  total peptide skipped by mod %d" % counter_skipped_mod)
  print("  total peptide skipped by len %d" % counter_skipped_len)
  print("  total peptide skipped by mass %d" % counter_skipped_mass)
  
#~ prepare_test_file(input_file="data/human.PXD002179.sds/peaks.db.mgf")#frac_1_"+str(frac)+".mgf")






# Partition a dbseq file into 2 sets: overlapping & nonoverlapping with the trainseq file
def partition_dbseq(dbseq_file, trainseq_file):
  
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
          
          if (peptide in trainseq):
            overlap_handle.write(line)
            count_overlap += 1
          else:
            nonoverlap_handle.write(line)
            count_nonoverlap += 1
          
          count += 1
  
  print("count = {0:d}".format(count))
  print("count_overlap = {0:d}".format(count_overlap))
  print("count_nonoverlap = {0:d}".format(count_nonoverlap))

#~ partition_dbseq(dbseq_file="data/human.cancer/peaks.db.frac_21_41.mgf.dbseq",
                #~ trainseq_file="data/human.cancer/peaks.db.frac_1_20.mgf.dbseq")
            
            

def read_dbseq(dbseq_file):
  
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
      #
      dbseq[scan] = [data_utils.vocab[x] for x in peptide]
      batch_len_AA += len(peptide)
  
  batch_size = len(dbseq)

  print("batch_size = ", batch_size)
  print("batch_len_AA = ", batch_len_AA)
  
  return dbseq, batch_size, batch_len_AA


def read_novonet(novonet_file):

  print("read_novonet()")
  print("novonet_file = ", novonet_file)

  novonet = {}

  with open(novonet_file, mode="r") as novonet_handle:
    
    # header
    novonet_handle.readline()
    
    for line in novonet_handle:
      
      line_split = re.split('\t|\n', line)
      scan = line_split[0]
      if (line_split[2] == ""): # empty output
        novonet_seq_id = []
      else:
        novonet_seq = re.split(',', line_split[2])
        novonet_seq_id = [data_utils.vocab[x] for x in novonet_seq]
      
      novonet[scan] = novonet_seq_id
      
  return novonet
  

def read_peaks(peaks_denovo_file, peaks_format, alc_threshold):

  print("read_peaks()")
  print("peaks_denovo_file = ", peaks_denovo_file)
  
  if (peaks_format == "old_7.5"):
    peptide_column = 1
    alc_column = 3
  elif (peaks_format == "new_8.0"):
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
      
      if (peaks_format == "old_7.5"):
        scan = line_split[0]
      elif (peaks_format == "new_8.0"):
        scan = "F" + line_split[0] + ":" + line_split[1]
        
      if (line_split[peptide_column] == ""): # empty output
        peaks_seq_id = []
      else:
        raw_sequence = line_split[peptide_column]
        raw_sequence_len = len(raw_sequence)
        peptide = []
        index = 0
        while (index < raw_sequence_len):
          if (raw_sequence[index]=="("):
            if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+57.02)"):
            #~ if (peptide[-1]=="C" and raw_sequence[index:index+8]=="(+58.01)"):
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
        peaks_seq_id = [data_utils.vocab[x] for x in peptide]

      alc_score = float(line_split[alc_column])
      if (alc_score >= alc_threshold):
        peaks[scan] = peaks_seq_id
        peaks_raw[scan] = raw_sequence
      
  return peaks, peaks_raw
  
def deepnovo_to_ALPS(deepnovo_file,output_file):
  
  print ("deepnovo to ALPS fille:", deepnovo_file)
    
  f = open(output_file,mode='w')
  f.write('Scan,Peptide,local confidence,total score, Area\n')

  
  with open(deepnovo_file, mode="r") as deepnovo_handle:
    
    # header
    deepnovo_handle.readline()
    
    for line in deepnovo_handle:
        line_split =  line.split('\t')
        scan = line_split[0]
        peptide = []
        #~ print (line)
        raw_sequence = line_split[2]
        raw_sequence = raw_sequence.split(",")
        raw_sequence_len = len(raw_sequence)
        if (raw_sequence_len >1): # not empty output      
          index = 0
          while (index < raw_sequence_len):
            if (raw_sequence[index]=="Cmod"):
              peptide.append("C(+57.02)")
            elif (raw_sequence[index]=="Mmod"):
              peptide.append("M(+15.99)") 
            elif (raw_sequence[index]=="Nmod"):
              peptide.append("N(+.98)")
            elif (raw_sequence[index]=="Qmod"):
              peptide.append("Q(+.98)")
            else:
              peptide.append(raw_sequence[index])
            index += 1
            
        alc_score = line_split[3]
        alc_score = alc_score.split(",")
        total_score = float(line_split[4])
        total_len = len(alc_score)
        # reason for using this formula: combine deepnovo with spider
        # output_score = exp(logProb)*1000
        # can change the number 1000 to any number else, to see the different result
        # using DeBruijn, it is int(np.exp
        output_aa_score = [float(np.exp(float(x))*100) for x in alc_score]
        
        # if only using result from deepnovo,
        # using DeBruijn_deepnovo,
        # the formula is:
        #~ output_aa_score = [float(x) for x in alc_score]
        #~ print (output_aa_score)
        
        pepstr = ''.join(peptide)
        #~ print ("peptide is:",peptide)
        f.write('%s,%s,%s,%f,1\n' %(scan,pepstr,' '.join(map(str,output_aa_score)),float(total_score/total_len)))
  f.close()
  
# now, the final score is divided by the length
deepnovo_to_ALPS(deepnovo_file="waters.mouse.light.da_4500.decode_output.tab",output_file="data/ab.testing/assem.waters.mouse.light/deepnovo_totalscore.csv")

def peaksdenovo_to_ALPS(peaksdenovo_file,output_file):
  
  print ("peaksdenovo to ALPS fille:", peaksdenovo_file)
    
  f = open(output_file,mode='w')
  f.write('Scan,Peptide,local confidence,Area\n')

  
  with open(peaksdenovo_file, mode="r") as peaksdenovo_handle:
    
    # header
    line = peaksdenovo_handle.readline()

    # beginning from the second line    
    for line in peaksdenovo_handle:
 
        line_split =  line.split(',')
        frac_number = line_split[0]
        scan_number = line_split[1]
        scan = "F"+frac_number+":"+scan_number
        peptide = line_split[3]
        alc_score = line_split[14]
        area = line_split[10]
        if (len(area)==0):
          area = 1
        f.write('%s,%s,%s,%s\n' %(scan,peptide,alc_score,area))
        
  f.close()

#~ peaksdenovo_to_ALPS(peaksdenovo_file="data/ab.testing/assem.public.mouse.heavy/peaks.denovo.csv",output_file="data/ab.testing/assem.public.mouse.heavy/ALPS.denovo.csv")        

def get_peaks_denovo_spectra(output_spectra_file, raw_spectra_file, peaks_denovo_file, peaks_format, alc_threshold=0):
  
  print("get_peaks_denovo_spectra()")
  print("peaks_denovo_file = ", peaks_denovo_file)
  print("ALC cut-off = ", alc_threshold)
  print("raw_spectra_file = ", raw_spectra_file)
  
  _, peaks_denovo_peptides = read_peaks(peaks_denovo_file, peaks_format, alc_threshold)
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
          if not (scan in peaks_denovo_peptides):
            continue
          else:
            counter_spectra += 1
            #
            for l in line_buffer:
              output_handle.write(l)
            #
            # RTINSECONDS
            line = input_handle.readline()
            output_handle.write(line)
            #
            # SEQ
            line = "SEQ=" + peaks_denovo_peptides[scan] + "\n"
            output_handle.write(line)
            #
            while line and not ("END IONS" in line):
              line = input_handle.readline()
              output_handle.write(line)
            #
            output_handle.write("\n")
              
        line = input_handle.readline()

  print("total spectra found %d" % counter_spectra)
  
  
#~ get_peaks_denovo_spectra(output_spectra_file="data/human.data2/peaks.denovo.alc_50.mgf",
                         #~ raw_spectra_file="data/human.data2/raw_spectra.mgf",
                         #~ peaks_denovo_file="data/human.data2/peaks.denovo.csv",
                         #~ peaks_format="old_7.5",
                         #~ alc_threshold=50)






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


def test_accuracy(dbseq_file, denovo_file, tool, peaks_format=None, alc_threshold=None):
  
  print("test_accuracy()")
  
  batch_accuracy_AA = 0.0
  batch_len_decode = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0

  dbseq, batch_size, batch_len_AA = read_dbseq(dbseq_file)
  
  if (tool == "novonet"):
    denovo = read_novonet(denovo_file)
  elif (tool == "peaks"):
    denovo, _ = read_peaks(denovo_file, peaks_format, alc_threshold)
  elif (tool =="novor"):
    denovo, _ = read_novor(denovo_file, peaks_format, alc_threshold)
    
  count_skipped = 0

  # for testing
  test_output = dict.fromkeys(dbseq.keys(),[])

  for scan, seq in denovo.iteritems():

    if (scan in dbseq):

      accuracy_AA = test_AA_match_novor(dbseq[scan], seq)
      len_AA = len(dbseq[scan])
      
      # for testing
      output_seq = [data_utils.vocab_reverse[x] for x in seq]
      test_output[scan] = [output_seq, accuracy_AA]
      
      len_decode = len(seq) 
      batch_len_decode += len_decode
      
      batch_accuracy_AA += accuracy_AA
      #~ batch_accuracy_AA += accuracy_AA/len_AA
      #
      if (accuracy_AA==len_AA):
        num_exact_match += 1.0
      #
      if (len(seq)==len_AA):
        num_len_match += 1.0
    
    else:
      
      count_skipped += 1
    
  # for testing
  with open("test_accuracy.tab", "w") as file_handle:
    file_handle.write("scan \t target_seq \t target_len \t output_seq \t accuracy_AA \n")
    for scan, output in test_output.iteritems():
      target_seq = [data_utils.vocab_reverse[x] for x in dbseq[scan]]
      target_len = len(target_seq)
      if (not output):
        file_handle.write("{0:s}\t{1:s}\t{2:d}\t{3:s}\t{4:d}\n".format(scan, target_seq, target_len, [], 0))
      else:
        file_handle.write("{0:s}\t{1:s}\t{2:d}\t{3:s}\t{4:d}\n".format(scan, target_seq, target_len, output[0], output[1]))

  print("  recall_AA %.4f" % (batch_accuracy_AA/batch_len_AA))

  #~ print("  accuracy_AA %.4f" % (batch_accuracy_AA/batch_size))
  print("  precision_AA %.4f" % (batch_accuracy_AA/batch_len_decode))
  #
  print("  recall_peptide %.4f" % (num_exact_match/batch_size))
  print("  recall_len %.4f" % (num_len_match/batch_size))
  print("  count_skipped (not in dbseq) %d" % (count_skipped))


# NovoNet
#~ test_accuracy(dbseq_file="data/yeast.full/peaks.db.frac_456.mgf.dbseq",
              #~ denovo_file = "train/train.intensity_only.yeast.full.db.frac_123.repeat/decode_output.db.tab",
              #~ tool="novonet")

#~ # PEAKS
#~ test_accuracy(dbseq_file="data/antibody.waters.mouse.HC/peaks.db.mgf.dbseq",
              #~ denovo_file = "novor_to_peaks.csv",
              #~ tool="peaks",
              #~ peaks_format="new_8.0",
              #~ alc_threshold=0)





