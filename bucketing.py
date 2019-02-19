# Copyright 2018 The Tensor2Tensor Authors.
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
"""Data reader module."""
import math
import fractions

import tensorflow as tf
import numpy as np


def _bucket_by_sequence_length(dataset,
                               example_length_fn,
                               bucket_boundaries,
                               bucket_batch_sizes):
  """Bucket entries in dataset by length.

  Args:
    dataset: Dataset of dict<feature name, Tensor>.
    example_length_fn: function from example to int, determines the length of
      the example, which will determine the bucket it goes into.
    bucket_boundaries: list<int>, boundaries of the buckets.
    bucket_batch_sizes: list<int>, batch size per bucket.

  Returns:
    Dataset of padded and batched examples.
  """
  with tf.name_scope("bucket_by_seq_length"):

    def example_to_bucket_id(example):
      """Return int64 id of the length bucket for this example."""
      seq_length = example_length_fn(example)

      boundaries = list(bucket_boundaries)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = tf.logical_and(
          tf.less_equal(buckets_min, seq_length),
          tf.less(seq_length, buckets_max))
      bucket_id = tf.reduce_min(tf.where(conditions_c))

      return bucket_id

    def window_size_fn(bucket_id):
      # window size = batch size
      batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
      window_size = batch_sizes[bucket_id]
      return window_size

    def batching_fn(bucket_id, grouped_dataset):
      batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
      batch_size = batch_sizes[bucket_id]
      return padded_batch(grouped_dataset, batch_size)

    dataset = dataset.apply(
        tf.data.experimental.group_by_window(example_to_bucket_id, batching_fn, None,
                                        window_size_fn))
    return dataset


def padded_batch(dataset, batch_size, padded_shapes=None):
  padded_shapes = padded_shapes or dict(
      [(name, [None] * len(shape))
       for name, shape in dataset.output_shapes.items()])
  return dataset.padded_batch(batch_size, padded_shapes)


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def _batching_scheme(batch_size,
                     max_length,
                     min_length_bucket,
                     length_bucket_step,
                     drop_long_sequences=False,
                     shard_multiplier=1,
                     length_multiplier=1,
                     min_length=0):
  """A batching scheme based on model hyperparameters.

  Every batch containins a number of sequences divisible by `shard_multiplier`.

  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.

  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example

  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3

  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret


def bucket_by_sequence_length(dataset, batch_size, example_length_fn):
    batching_scheme = _batching_scheme(
            batch_size=batch_size,
            max_length=0,
            min_length_bucket=8,
            length_bucket_step=1.1)

    dataset = _bucket_by_sequence_length(
            dataset,
            example_length_fn,
            batching_scheme["boundaries"],
            batching_scheme["batch_sizes"])

    return dataset
