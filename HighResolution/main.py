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

import data_utils
import translate_caption_2dir






def main(_):
  print("main()")
  
  if (data_utils.FLAGS.knapsack_build):
    translate_caption_2dir.knapsack_build()
  elif (data_utils.FLAGS.train):
    translate_caption_2dir.train()
  elif (data_utils.FLAGS.test_true_feeding):
    translate_caption_2dir.test_true_feeding()
  elif (data_utils.FLAGS.decode):
    translate_caption_2dir.decode()
  else:
    print("ERROR: wrong option!")
    sys.exit()


if __name__ == "__main__":
  tf.app.run()
