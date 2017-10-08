# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import deepnovo_config
from deepnovo_cython_modules import process_spectrum


class WorkerIO(object):
  """TODO(nh2tran): docstring.
  """


  def __init__(self, input_file, output_file=None):
    """TODO(nh2tran): docstring.
       The input_file could be input_file or input_file_train/valid/test.
       The output_file is None for train/valid/test cases.
       During training we use two separate WorkerIO objects for train and valid.
    """

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.__init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.MZ_MAX = deepnovo_config.MZ_MAX
    self.batch_size = deepnovo_config.batch_size

    self.input_file = input_file
    self.output_file = output_file
    print("input_file = {0:s}".format(self.input_file))
    print("output_file = {0:s}".format(self.output_file))
    # keep the file handles open throughout the process to read/write batches
    self.input_handle = None
    self.output_handle = None

    # store the file location of all spectra for random access
    self.location_list = []
    # split data into batches
    self.location_batch_list = []
    self.location_batch_count = 0

    # record the status of spectra that have been read
    self.spectrum_count = {"total": 0,
                           "read": 0,
                           "skipped": 0,
                           "skipped_mass": 0}


  def close_input(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.close_input()")

    self.input_handle.close()


  def close_output(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.close_output()")

    self.output_handle.close()


  def get_spectrum(self, location_batch):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO.get_spectrum()")

    spectrum_list = []
    for location in location_batch:

      # parse a spectrum
      (precursor_mz,
       charge,
       scan,
       raw_sequence,
       mz_list,
       intensity_list) = self._parse_spectrum(location)

      # skip if precursor_mass > MZ_MAX
      precursor_mass = precursor_mz * charge - deepnovo_config.mass_H * charge
      if precursor_mass > self.MZ_MAX:
        self.spectrum_count["skipped"] += 1
        self.spectrum_count["skipped_mass"] += 1
        continue
      self.spectrum_count["read"] += 1

      # pre-process spectrum
      (spectrum_holder,
       spectrum_original_forward,
       spectrum_original_backward) = process_spectrum(mz_list,
                                                      intensity_list,
                                                      precursor_mass)

      # update dataset
      spectrum = {"scan": scan,
                  "precursor_mass": precursor_mass,
                  "spectrum_holder": spectrum_holder,
                  "spectrum_original_forward": spectrum_original_forward,
                  "spectrum_original_backward": spectrum_original_backward}
      spectrum_list.append(spectrum)

    return spectrum_list


  def get_location(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.get_location()")

    location_list = []
    keyword = "BEGIN IONS"
    line = True
    while line:
      location = self.input_handle.tell()
      line = self.input_handle.readline()
      if keyword in line:
        location_list.append(location)

    self.location_list = location_list
    self.spectrum_count["total"] = len(location_list)


  def open_input(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.open_input()")

    self.input_handle = open(self.input_file, 'r')


  def open_output(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.open_output()")

    self.output_handle = open(self.output_file, 'w')
    self._print_prediction_header()


  def split_location(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO.split_location()")

    location_batch_list = [self.location_list[i:(i+self.batch_size)]
                            for i in range(0,
                                           self.spectrum_count["total"],
                                           self.batch_size)]

    self.location_batch_list = location_batch_list
    self.location_batch_count = len(self.location_batch_list)


  def write_prediction(self, predicted_batch):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO.write_prediction()")

    for predicted in predicted_batch:
      scan = predicted["scan"]
      predicted_sequence = ",".join(predicted["sequence"])
      predicted_score = "{0:.2f}".format(predicted["score"])
      predicted_position_score = ",".join([
          "{0:.2f}".format(x) for x in predicted["position_score"]])
      predicted_row = "\t".join([scan,
                                 predicted_sequence,
                                 predicted_score,
                                 predicted_position_score])
      print(predicted_row, file=self.output_handle, end="\n")


  def _parse_spectrum(self, location):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO._parse_spectrum()")

    self.input_handle.seek(location)
    # BEGIN IONS
    line = self.input_handle.readline()
    precursor_mz, charge, scan, raw_sequence = self._parse_spectrum_header()
    mz_list, intensity_list = self._parse_spectrum_ion()

    return precursor_mz, charge, scan, raw_sequence, mz_list, intensity_list


  def _parse_spectrum_header(self):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO._parse_spectrum_header()")

    # header TITLE
    line = self.input_handle.readline()
    # header PEPMASS
    line = self.input_handle.readline()
    precursor_mz = float(re.split('=|\n', line)[1])
    # header CHARGE
    line = self.input_handle.readline()
    charge = float(re.split('=|\+', line)[1])
    # header SCANS
    line = self.input_handle.readline()
    scan = re.split('=|\n', line)[1]
    # header RTINSECONDS
    line = self.input_handle.readline()
    # header SEQ
    line = self.input_handle.readline()
    raw_sequence = re.split('=|\n', line)[1]

    return precursor_mz, charge, scan, raw_sequence


  def _parse_spectrum_ion(self):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO._parse_spectrum_ion()")

    # ion
    mz_list = []
    intensity_list = []
    line = self.input_handle.readline()
    while not "END IONS" in line:
      mz, intensity = re.split(' |\n', line)[:2]
      mz_float = float(mz)
      intensity_float = float(intensity)
      # skip an ion if its mass > MZ_MAX
      if mz_float > self.MZ_MAX:
        line = self.input_handle.readline()
        continue
      mz_list.append(mz_float)
      intensity_list.append(intensity_float)
      line = self.input_handle.readline()

    return mz_list, intensity_list


  def _print_prediction_header(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO._print_prediction_header()")

    header_list = ["scan",
                   "predicted_sequence",
                   "predicted_score",
                   "predicted_position_score"]
    header_row = "\t".join(header_list)
    print(header_row, file=self.output_handle, end="\n")

