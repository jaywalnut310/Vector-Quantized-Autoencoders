import sys
import tensorflow as tf
import bucketing


class Vocabulary:
  PAD_TOKEN = "<PAD>"
  START_TOKEN = "<START>"
  END_TOKEN = "<END>"
  UNK_TOKEN = "<UNK>"
  RESERVED_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
  PAD_TOKEN_ID = RESERVED_TOKENS.index(PAD_TOKEN)
  START_TOKEN_ID = RESERVED_TOKENS.index(START_TOKEN)
  END_TOKEN_ID = RESERVED_TOKENS.index(END_TOKEN)
  UNK_TOKEN_ID = RESERVED_TOKENS.index(UNK_TOKEN)

  def __init__(self, filename):
    rows = [line.strip().split("\t") for line in open(filename)]    # [(token, vector), ...]
    self.tokens = Vocabulary.RESERVED_TOKENS + [r[0] for r in rows]
    self.token_to_id_map = {t: i for i, t in enumerate(self.tokens)}

  def token_to_id(self, token):
    return self.token_to_id_map.get(token, Vocabulary.UNK_TOKEN_ID)

  def id_to_token(self, id_):
    return self.tokens[id_]

  def encode(self, s, append_eos=True):
    ids = [self.token_to_id(t) for t in s.split()]
    if append_eos:
        ids += [Vocabulary.END_TOKEN_ID]
    return ids

  def decode(self, ids, truncate=True):
    if truncate:
        ids = Vocabulary.truncate(ids)
    tokens = [self.id_to_token(id_) for id_ in ids]
    return " ".join(tokens)

  def __len__(self):
    return len(self.tokens)

  @classmethod
  def truncate(cls, ids):
    try:
      pos = ids.index(cls.END_TOKEN_ID)
    except ValueError:
      return ids
    else:
      return ids[:pos]


def load_vocab(hparams):
  hparams.source_vocab = Vocabulary(hparams.source_vocab_file)
  if hparams.target_vocab_file == hparams.source_vocab_file:
    hparams.target_vocab = hparams.source_vocab
  else:
    hparams.target_vocab = Vocabulary(hparams.target_vocab_file)


def example_length(example):
  return tf.maximum(tf.size(example["source"]), tf.size(example["target"]))


def parse_record(example):
  features = tf.parse_single_example(example, features={
    'source': tf.VarLenFeature(tf.int64),
    'target': tf.VarLenFeature(tf.int64)})
  features = {
    'source': tf.sparse.to_dense(features['source']),
    'target': tf.sparse.to_dense(features['target']),}
  return features

class InputPipeline:
  def __init__(self, source_file, target_file, record_file, mode, hparams):
    self.source_file = source_file
    self.target_file = target_file
    self.record_file = record_file
    self.mode = mode
    self.hparams = hparams

  def _readlines(self):
    if self.mode != tf.estimator.ModeKeys.PREDICT:
      for s, t in zip(open(self.source_file), open(self.target_file)):
        source = self.hparams.source_vocab.encode(s.strip())
        target = self.hparams.target_vocab.encode(t.strip())
        yield {'source': source, 'target': target}
    else:
      for s in open(self.source_file):
        source = self.hparams.source_vocab.encode(s.strip())
        yield {'source': source, 'target': []}

  def _postprocess(self, features):
    final_features = {
      "sources": features["source"],
      "targets": features["target"],
    }
    return final_features, {}

  def __call__(self):
    hparams = self.hparams
    with tf.name_scope("input_pipeline"):
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        dataset = tf.data.TFRecordDataset(self.record_file)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(parse_record, num_parallel_calls=8)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
          dataset = dataset.filter(lambda example: tf.logical_and(
              tf.less_equal(tf.size(example["source"]), hparams.max_input_length),
              tf.less_equal(tf.size(example["target"]), hparams.max_input_length)))
        dataset = bucketing.bucket_by_sequence_length(
                dataset,
                hparams.batch_size,
                example_length_fn=example_length)
        dataset = dataset.map(
                self._postprocess,
                num_parallel_calls=8)
        dataset = dataset.prefetch(buffer_size=1)
        
        return dataset
      else:
        dataset = tf.data.Dataset.from_generator(
            self._readlines,
            {'source': tf.int32, 'target': tf.int32},
            {'source': [None], 'target': [None]})

        dataset = bucketing.padded_batch(
                dataset, hparams.predict_batch_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        final_features = {
            "sources": features["source"],
            "targets": None,
        }

        return final_features, None
