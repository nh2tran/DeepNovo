from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

import tensorflow as tf

import data_utils






def get_candidate_intensity(spectrum_original, peptide_mass, prefix_mass, direction):

  # a-ions
#~   candidate_a_mass = prefix_mass + data_utils.mass_ID_np - data_utils.mass_CO
#~   candidate_a_H2O = candidate_a_mass - data_utils.mass_H2O
#~   candidate_a_NH3 = candidate_a_mass - data_utils.mass_NH3
#~   candidate_a_plus2_charge1 = (candidate_a_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
#~   a_ions = [candidate_a_mass,candidate_a_H2O,candidate_a_NH3,candidate_a_plus2_charge1]
 
  # x-ions
#~   candidate_x_mass = peptide_mass - candidate_a_mass
#~   candidate_x_H2O = candidate_x_mass - data_utils.mass_H2O
#~   candidate_x_NH3 = candidate_x_mass - data_utils.mass_NH3
#~   candidate_x_plus2_charge1 = (candidate_x_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
#~   x_ions = [candidate_x_mass,candidate_x_H2O,candidate_x_NH3,candidate_x_plus2_charge1]
 
  # b-ions
  candidate_b_mass = prefix_mass + data_utils.mass_ID_np 
#~   b_ions = [candidate_b_mass]
  candidate_b_H2O = candidate_b_mass - data_utils.mass_H2O
  candidate_b_NH3 = candidate_b_mass - data_utils.mass_NH3
  candidate_b_plus2_charge1 = (candidate_b_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
  b_ions = [candidate_b_mass,candidate_b_H2O,candidate_b_NH3,candidate_b_plus2_charge1]

  # y-ions
  candidate_y_mass = peptide_mass - candidate_b_mass 
#~   y_ions = [candidate_y_mass] 
  candidate_y_H2O = candidate_y_mass - data_utils.mass_H2O
  candidate_y_NH3 = candidate_y_mass - data_utils.mass_NH3
  candidate_y_plus2_charge1 = (candidate_y_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
  y_ions = [candidate_y_mass,candidate_y_H2O,candidate_y_NH3,candidate_y_plus2_charge1]
  
  # c-ions
#~   candidate_c_mass = prefix_mass + data_utils.mass_ID_np + data_utils.mass_NH3
#~   candidate_c_H2O = candidate_c_mass - data_utils.mass_H2O
#~   candidate_c_NH3 = candidate_c_mass - data_utils.mass_NH3
#~   candidate_c_plus2_charge1 = (candidate_c_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
#~   c_ions = [candidate_c_mass,candidate_c_H2O,candidate_c_NH3,candidate_c_plus2_charge1]

  # z-ions
#~   candidate_z_mass = peptide_mass - candidate_c_mass
#~   candidate_z_H2O = candidate_z_mass - data_utils.mass_H2O
#~   candidate_z_NH3 = candidate_z_mass - data_utils.mass_NH3
#~   candidate_z_plus2_charge1 = (candidate_z_mass + 2 * data_utils.mass_H) / 2 - data_utils.mass_H
#~   z_ions = [candidate_z_mass,candidate_z_H2O,candidate_z_NH3,candidate_y_plus2_charge1]
  
  # ion_mass_list & FIRST_LABEL
  if (direction==0):
    FIRST_LABEL = data_utils.GO_ID
    LAST_LABEL = data_utils.EOS_ID
    ion_mass_list = b_ions + y_ions
#~     ion_mass_list = a_ions + b_ions + c_ions + x_ions + y_ions + z_ions
  elif (direction==1):
    FIRST_LABEL = data_utils.EOS_ID
    LAST_LABEL = data_utils.GO_ID
    ion_mass_list = y_ions + b_ions
#~     ion_mass_list = z_ions + y_ions + x_ions + c_ions + b_ions + a_ions
  #
  ion_mass = np.array(ion_mass_list, dtype=np.float32)

  # ion locations
  location_sub50 = np.rint(ion_mass*data_utils.RESOLUTION).astype(np.int32)
  location_sub50 -= data_utils.RESOLUTION_HALF
  #
  location_plus50 = location_sub50 + data_utils.RESOLUTION
  #
  ion_id_rows, aa_id_cols = np.nonzero(np.logical_and(
                                          location_sub50 >= 0,
                                          location_plus50 <= data_utils.MZ_SIZE))

  # candidate_intensity
  candidate_intensity = np.zeros(shape=(data_utils.vocab_size,
                                        data_utils.num_ion,
                                        data_utils.RESOLUTION),
                                        dtype=np.float32)
  #
  cdef int [:,:] location_sub50_view = location_sub50
  cdef int [:,:] location_plus50_view = location_plus50
  cdef float [:,:,:] candidate_intensity_view = candidate_intensity
  cdef float [:] spectrum_original_view = spectrum_original
  cdef int[:] row = ion_id_rows.astype(np.int32)
  cdef int[:] col = aa_id_cols.astype(np.int32)
  cdef int index
  for index in xrange(ion_id_rows.size):
    candidate_intensity_view[col[index],row[index],:] = spectrum_original_view[location_sub50_view[row[index],col[index]]:location_plus50_view[row[index],col[index]]]

  # PAD/GO/EOS
  candidate_intensity[data_utils.PAD_ID].fill(0.0)
  #
  candidate_intensity[FIRST_LABEL].fill(0.0)
  #
  candidate_intensity[LAST_LABEL].fill(0.0)
  #~ b_ion_count = len(b_ions)
  #~ if (direction==0):
    #~ candidate_intensity[LAST_LABEL,b_ion_count:].fill(0.0)
  #~ elif (direction==1):
    #~ candidate_intensity[LAST_LABEL,:b_ion_count].fill(0.0)
    
  #~ for aa_id in ([LAST_LABEL] + range(3,data_utils.vocab_size)):
    #~ for ion_id in xrange(data_utils.num_ion):
      #~ location_sub50 = location_sub50_list[ion_id][aa_id]
      #~ #
      #~ if (location_sub50 > 0):
        #~ candidate_intensity[aa_id,ion_id] = spectrum_original[location_sub50:location_sub50+data_utils.RESOLUTION]
        
  # Nomalization to N(0,1): tf.image.per_image_whitening
#~   adjusted_stddev = max(np.std(candidate_intensity), 1.0/math.sqrt(candidate_intensity.size))
#~   candidate_intensity = (candidate_intensity-np.mean(candidate_intensity)) / adjusted_stddev

  return candidate_intensity


def process_spectrum(spectrum_mz_list, spectrum_intensity_list, peptide_mass):

  # neutral mass, location, normalized intensity, assuming ion charge z=1
  #
  charge = 1.0
  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  neutral_mass = spectrum_mz - charge*data_utils.mass_H
  neutral_mass_location = np.rint(neutral_mass*data_utils.RESOLUTION).astype(np.int32)
  cdef int [:] neutral_mass_location_view = neutral_mass_location
  #
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  norm_intensity = spectrum_intensity / np.max(spectrum_intensity)
  cdef float [:] norm_intensity_view = norm_intensity
  
  # fill spectrum holders
  #
  spectrum_holder = np.zeros(shape=data_utils.MZ_SIZE, dtype=np.float32)
  cdef float [:] spectrum_holder_view = spectrum_holder
  #
  # note that different peaks may fall into the same location, hence loop +=
  cdef int index
  for index in xrange(neutral_mass_location.size):
    #
#~     spectrum_holder_view[neutral_mass_location_view[index]] += norm_intensity_view[index]
    spectrum_holder_view[neutral_mass_location_view[index]] = max(spectrum_holder_view[neutral_mass_location_view[index]], 
                                                                  norm_intensity_view[index])
  #
  spectrum_original_forward = np.copy(spectrum_holder)
  #
  spectrum_original_backward = np.copy(spectrum_holder)

  # add complement
  complement_mass = peptide_mass - neutral_mass
  complement_mass_location = np.rint(complement_mass*data_utils.RESOLUTION).astype(np.int32)
  cdef int [:] complement_mass_location_view = complement_mass_location
  #
#~   cdef int index
  for index in np.nonzero(complement_mass_location>0)[0]:
    spectrum_holder_view[complement_mass_location_view[index]] += norm_intensity_view[index]
  

  
  
  
  
  # peptide_mass
  #
  spectrum_original_forward[int(round(peptide_mass*data_utils.RESOLUTION))] = 1.0
  #
  spectrum_original_backward[int(round(peptide_mass*data_utils.RESOLUTION))] = 1.0

  # N-terminal, b-ion, peptide_mass_C
  #
  # append N-terminal
  mass_N = data_utils.mass_N_terminus - data_utils.mass_H
  spectrum_holder[int(round(mass_N*data_utils.RESOLUTION))] = 1.0
  #
  # append peptide_mass_C
  mass_C = data_utils.mass_C_terminus + data_utils.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_holder[int(round(peptide_mass_C*data_utils.RESOLUTION))] = 1.0
  #
  spectrum_original_forward[int(round(peptide_mass_C*data_utils.RESOLUTION))] = 1.0

  # C-terminal, y-ion, peptide_mass_N
  #
  # append C-terminal
  mass_C = data_utils.mass_C_terminus + data_utils.mass_H
  spectrum_holder[int(round(mass_C*data_utils.RESOLUTION))] = 1.0
  #
  # append peptide_mass_N
  mass_N = data_utils.mass_N_terminus - data_utils.mass_H
  peptide_mass_N = peptide_mass - mass_N
  spectrum_holder[int(round(peptide_mass_N*data_utils.RESOLUTION))] = 1.0
  #
  spectrum_original_backward[int(round(peptide_mass_N*data_utils.RESOLUTION))] = 1.0

  return spectrum_holder, spectrum_original_forward, spectrum_original_backward

      

