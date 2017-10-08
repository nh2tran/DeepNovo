# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepnovo_config


def get_candidate_intensity(spectrum_original,
                            peptide_mass,
                            prefix_mass,
                            direction):
  """TODO(nh2tran): docstring."""

  # FIRST_LABEL & prefix_mass
  if direction == 0:
    FIRST_LABEL = deepnovo_config.GO_ID
    LAST_LABEL = deepnovo_config.EOS_ID
    candidate_b_mass = prefix_mass + deepnovo_config.mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    FIRST_LABEL = deepnovo_config.EOS_ID
    LAST_LABEL = deepnovo_config.GO_ID
    candidate_y_mass = prefix_mass + deepnovo_config.mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass

  # b-ions
  candidate_b_H2O = candidate_b_mass - deepnovo_config.mass_H2O
  candidate_b_NH3 = candidate_b_mass - deepnovo_config.mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * deepnovo_config.mass_H) / 2
                               - deepnovo_config.mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - deepnovo_config.mass_H2O
  candidate_y_NH3 = candidate_y_mass - deepnovo_config.mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * deepnovo_config.mass_H) / 2
                               - deepnovo_config.mass_H)

  # ion_2
#~   b_ions = [candidate_b_mass]
#~   y_ions = [candidate_y_mass]
#~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  ion_mass_list = b_ions + y_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)

  # ion locations
  location_sub50 = np.rint(ion_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  location_sub50 -= (deepnovo_config.WINDOW_SIZE // 2)
  location_plus50 = location_sub50 + deepnovo_config.WINDOW_SIZE
  ion_id_rows, aa_id_cols = np.nonzero(np.logical_and(
      location_sub50 >= 0,
      location_plus50 <= deepnovo_config.MZ_SIZE))

  # candidate_intensity
  candidate_intensity = np.zeros(shape=(deepnovo_config.vocab_size,
                                        deepnovo_config.num_ion,
                                        deepnovo_config.WINDOW_SIZE),
                                 dtype=np.float32)
  cdef int [:,:] location_sub50_view = location_sub50
  cdef int [:,:] location_plus50_view = location_plus50
  cdef float [:,:,:] candidate_intensity_view = candidate_intensity
  cdef float [:] spectrum_original_view = spectrum_original
  cdef int[:] row = ion_id_rows.astype(np.int32)
  cdef int[:] col = aa_id_cols.astype(np.int32)
  cdef int index
  for index in xrange(ion_id_rows.size):
    candidate_intensity_view[col[index], row[index], :] = spectrum_original_view[location_sub50_view[row[index], col[index]] : location_plus50_view[row[index], col[index]]] # TODO(nh2tran): line-too-long

  # PAD/GO/EOS
  candidate_intensity[deepnovo_config.PAD_ID].fill(0.0)
  candidate_intensity[FIRST_LABEL].fill(0.0)
  candidate_intensity[LAST_LABEL].fill(0.0)
  #~ b_ion_count = len(b_ions)
  #~ if (direction==0):
    #~ candidate_intensity[LAST_LABEL,b_ion_count:].fill(0.0)
  #~ elif (direction==1):
    #~ candidate_intensity[LAST_LABEL,:b_ion_count].fill(0.0)

  #~ for aa_id in ([LAST_LABEL] + range(3,deepnovo_config.vocab_size)):
    #~ for ion_id in xrange(deepnovo_config.num_ion):
      #~ location_sub50 = location_sub50_list[ion_id][aa_id]
      #~ #
      #~ if (location_sub50 > 0):
        #~ candidate_intensity[aa_id,ion_id] = spectrum_original[location_sub50:location_sub50+deepnovo_config.WINDOW_SIZE]

  # Nomalization to N(0,1): tf.image.per_image_whitening
#~   adjusted_stddev = max(np.std(candidate_intensity), 1.0/math.sqrt(candidate_intensity.size))
#~   candidate_intensity = (candidate_intensity-np.mean(candidate_intensity)) / adjusted_stddev

  return candidate_intensity


def process_spectrum(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """TODO(nh2tran): docstring."""

  # neutral mass, location, assuming ion charge z=1
  charge = 1.0
  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  neutral_mass_location = np.rint(neutral_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  cdef int [:] neutral_mass_location_view = neutral_mass_location

  # normalize intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  norm_intensity = spectrum_intensity / np.max(spectrum_intensity)
  cdef float [:] norm_intensity_view = norm_intensity

  # fill spectrum holders
  spectrum_holder = np.zeros(shape=deepnovo_config.MZ_SIZE, dtype=np.float32)
  cdef float [:] spectrum_holder_view = spectrum_holder
  # note that different peaks may fall into the same location, hence loop +=
  cdef int index
  for index in xrange(neutral_mass_location.size):
#~     spectrum_holder_view[neutral_mass_location_view[index]] += norm_intensity_view[index] # TODO(nh2tran): line-too-long
    spectrum_holder_view[neutral_mass_location_view[index]] = max(spectrum_holder_view[neutral_mass_location_view[index]], # TODO(nh2tran): line-too-long
                                                                  norm_intensity_view[index]) # TODO(nh2tran): line-too-long
  spectrum_original_forward = np.copy(spectrum_holder)
  spectrum_original_backward = np.copy(spectrum_holder)

  # add complement
  complement_mass = peptide_mass - neutral_mass
  complement_mass_location = np.rint(complement_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  cdef int [:] complement_mass_location_view = complement_mass_location
#~   cdef int index
  for index in np.nonzero(complement_mass_location > 0)[0]:
    spectrum_holder_view[complement_mass_location_view[index]] += norm_intensity_view[index] # TODO(nh2tran): line-too-long

  # peptide_mass
  spectrum_original_forward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long
  spectrum_original_backward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_holder[int(round(mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_holder[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long
  spectrum_original_forward[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_holder[int(round(mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_N
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  peptide_mass_N = peptide_mass - mass_N
  spectrum_holder[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long
  spectrum_original_backward[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0 # TODO(nh2tran): line-too-long

  return spectrum_holder, spectrum_original_forward, spectrum_original_backward
