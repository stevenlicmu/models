# Lint as: python3
# Copyright 2020 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Performs DELG-based image retrieval on Revisited Oxford/Paris datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from delf import datum_io
from delf.python.detect_to_retrieve import dataset
from delf.python.detect_to_retrieve import image_reranking

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_file_path', '/tmp/gnd_roxford5k.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string('query_features_dir', '/tmp/features/query',
                    'Directory where query DELG features are located.')
flags.DEFINE_string('index_features_dir', '/tmp/features/index',
                    'Directory where index DELG features are located.')
flags.DEFINE_boolean(
    'use_geometric_verification', False,
    'If True, performs re-ranking using local feature-based geometric '
    'verification.')
flags.DEFINE_float(
    'local_descriptor_matching_threshold', 1.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Threshold below which a pair of local descriptors is considered '
    'a potential match, and will be fed into RANSAC.')
flags.DEFINE_float(
    'ransac_residual_threshold', 20.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Residual error threshold for considering matches as inliers, used in '
    'RANSAC algorithm.')
flags.DEFINE_boolean(
    'use_ratio_test', False,
    'Optional, only used if `use_geometric_verification` is True. '
    'Whether to use ratio test for local feature matching.')
flags.DEFINE_string(
    'output_dir', '/tmp/retrieval',
    'Directory where retrieval output will be written to. A file containing '
    "metrics for this run is saved therein, with file name 'metrics.txt'.")

# Extensions.
_DELG_GLOBAL_EXTENSION = '.delg_global'
_DELG_LOCAL_EXTENSION = '.delg_local'

# Precision-recall ranks to use in metric computation.
_PR_RANKS = (1, 5, 10)

# Pace to log.
_STATUS_CHECK_LOAD_ITERATIONS = 50

# Output file names.
_METRICS_FILENAME = 'metrics.txt'

_COLUMN_TO_NAME = {0: 't2', 1: 't3', 2: 't4', 3: 't5', 4: 'y2', 5: 'y3', 6: 'y4', 7: 'y5', 8: 'z2', 9: 'z3', 10: 'z4', 11: 'z5', 12: 'x2', 13: 'x3', 14: 'x4', 15: 'x5', 16: 'x6', 17: 'j2', 18: 'j3', 19: 'j4', 20: 'j5', 21: 'j6'}

def _ReadDelgGlobalDescriptors(input_dir, image_list):
  """Reads DELG global features.

  Args:
    input_dir: Directory where features are located.
    image_list: List of image names for which to load features.

  Returns:
    global_descriptors: NumPy array of shape (len(image_list), D), where D
      corresponds to the global descriptor dimensionality.
  """
  num_images = len(image_list)
  global_descriptors = []
  print('Starting to collect global descriptors for %d images...' % num_images)
  start = time.time()
  for i in range(num_images):
    if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Reading global descriptors for image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
      start = time.time()

    descriptor_filename = image_list[i] + _DELG_GLOBAL_EXTENSION
    descriptor_fullpath = os.path.join(input_dir, descriptor_filename)
    global_descriptors.append(datum_io.ReadFromFile(descriptor_fullpath))

  return np.array(global_descriptors)

def map_column_to_name(column):
  return _COLUMN_TO_NAME.get(column, "not_found")

def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  query_list = ['t1', 'y1', 'z1', 'x1', 'j1']
  index_list = ['t2', 't3', 't4', 't5', 'y2', 'y3', 'y4', 'y5', 'z2', 'z3', 'z4', 'z5', 'x2', 'x3', 'x4', 'x5', 'x6', 'j2', 'j3', 'j4', 'j5', 'j6']
  num_query_images = len(query_list)
  num_index_images = len(index_list)
  print(f'here is the query list: {query_list}')
  print(f'here is the index list: {index_list}')

  print('done! Found %d queries and %d index images' %
        (num_query_images, num_index_images))

  # Read global features.
  query_global_features = _ReadDelgGlobalDescriptors(FLAGS.query_features_dir,
                                                     query_list)
  index_global_features = _ReadDelgGlobalDescriptors(FLAGS.index_features_dir,
                                                     index_list)

  # Compute similarity between query and index images, potentially re-ranking
  # with geometric verification.
  ranks_before_gv = np.zeros([num_query_images, num_index_images],
                             dtype='int32')
  ranks_sim_before_gv = []
  if FLAGS.use_geometric_verification:
    medium_ranks_after_gv = np.zeros([num_query_images, num_index_images],
                                     dtype='int32')
  for i in range(num_query_images):
    print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
    start = time.time()

    # Compute similarity between global descriptors.
    similarities = np.dot(index_global_features, query_global_features[i])
    # print(f'similarities matrix is: {similarities}')
    ranks_before_gv[i] = np.argsort(-similarities)
    ranks_sim_before_gv_row = [f'{_COLUMN_TO_NAME.get(index, "None")}: {similarities[index]}' for index in ranks_before_gv[i]]
    # ranks_sim_before_gv.append(ranks_sim_before_gv_row)
    print(f'similarities with file name: {ranks_sim_before_gv_row}')

    # Re-rank using geometric verification.
    if FLAGS.use_geometric_verification:
      print('Performing local feature match: ')
      medium_ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          input_ranks=ranks_before_gv[i],
          initial_scores=similarities,
          query_name=query_list[i],
          index_names=index_list,
          query_features_dir=FLAGS.query_features_dir,
          index_features_dir=FLAGS.index_features_dir,
          local_feature_extension=_DELG_LOCAL_EXTENSION,
          ransac_seed=0,
          descriptor_matching_threshold=FLAGS
          .local_descriptor_matching_threshold,
          ransac_residual_threshold=FLAGS.ransac_residual_threshold,
          use_ratio_test=FLAGS.use_ratio_test)

    elapsed = (time.time() - start)
    print('done! Retrieval for query %d took %f seconds' % (i, elapsed))

  
  print(f'ranks_before_gv: {ranks_before_gv}')
  # print(f'ranks_with_similarity_before_gv: {ranks_sim_before_gv}')

  # print(f'ranks_after_gv: {medium_ranks_after_gv}')

if __name__ == '__main__':
  app.run(main)
