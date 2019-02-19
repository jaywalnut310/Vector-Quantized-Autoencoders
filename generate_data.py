import os
import random
import math
import argparse
from typing import Iterable


import tensorflow as tf

from hparams import Hparams, import_configs
import data

"""
Generate text datasets
"""

def _int64_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_tfrecords(source_file, target_file, out_file, source_vocab, target_vocab):
  with tf.python_io.TFRecordWriter(out_file) as writer:
    for count, (s, t) in enumerate(zip(open(source_file, 'r'), open(target_file, 'r'))):
      source = source_vocab.encode(s.strip())
      target = target_vocab.encode(t.strip())

      feature = {'source': _int64_feature(source),
                 'target': _int64_feature(target)}

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())

      if count % 100000 == 0:
        tf.logging.info("write: %d", count)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Generate and save datasets")
  # Parser must contain 8 key-value pairs:
  # source_vocab_file, target_vocab_file,
  # source_train_file, target_train_file
  # source_eval_file, target_eval_file
  # record_train_file, record_eval_file
  parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
  args = parser.parse_args()

  hparams = Hparams()
  import_configs(hparams, args.configs)
  
  # Prepare data
  data.load_vocab(hparams)

  # Save tfrecords
  tf.logging.set_verbosity(tf.logging.INFO)
  save_tfrecords(hparams.source_train_file, 
                 hparams.target_train_file, 
                 hparams.record_train_file, 
                 hparams.source_vocab,
                 hparams.target_vocab)
  save_tfrecords(hparams.source_eval_file, 
                 hparams.target_eval_file, 
                 hparams.record_eval_file, 
                 hparams.source_vocab,
                 hparams.target_vocab)
