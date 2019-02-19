import os
import math
import functools
import tensorflow as tf
from tensorflow.python.training import distribution_strategy_context

import commons
import transformer
import bleu
import beam_search


def init_vq_bottleneck(bottleneck_size, hidden_size, mean_only=False):
  """Get lookup table for VQ bottleneck."""
  means = tf.get_variable(
      name="means",
      shape=[bottleneck_size, hidden_size],
      initializer=tf.initializers.variance_scaling(distribution="uniform"))
  if not mean_only:
    ema_count = tf.get_variable(
        name="ema_count",
        shape=[bottleneck_size],
        initializer=tf.constant_initializer(0),
        trainable=False)
    ema_means = tf.get_variable(
        name="ema_means",
        initializer=means.initialized_value(),
        trainable=False)
  else:
    ema_count = None
    ema_means = None

  return means, ema_means, ema_count


def vq_nearest_neighbor(x, hparams):
  """Find the nearest element in means to elements in x."""
  bottleneck_size = 2**hparams.bottleneck_bits
  means = hparams.means
  x_sg = tf.stop_gradient(x)
  x_norm_sq = tf.reduce_sum(tf.square(x_sg), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  scalar_prod = tf.matmul(x_sg, means, transpose_b=True)
  dist = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod
  if hparams.bottleneck_kind in ["em" ,"mog"]:
    x_means_idx = tf.multinomial(-dist, num_samples=hparams.num_samples)
    x_means_hot = tf.one_hot(
        x_means_idx, depth=bottleneck_size)
    x_means_hot = tf.reduce_mean(x_means_hot, axis=1)
  else:
    x_means_idx = tf.argmax(-dist, axis=-1)
    x_means_hot = tf.one_hot(x_means_idx, depth=bottleneck_size)
  x_means = tf.matmul(x_means_hot, means)
  e_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(x_means)))

  if hparams.bottleneck_kind == "mog":
    m_shape = [tf.to_float(x) for x in commons.shape_list(means)]
    logp = -tf.log(m_shape[0]) - .5 * tf.log(2 * math.pi) * m_shape[1] + tf.reduce_logsumexp(-.5 * dist, -1)
    e_loss -= hparams.gamma * tf.reduce_mean(logp)
  return x_means_hot, e_loss


def vq_discrete_bottleneck(x, hparams):
  """Simple vector quantized discrete bottleneck."""
  bottleneck_size = 2**hparams.bottleneck_bits
  x_shape = commons.shape_list(x)
  x = tf.reshape(x, [-1, hparams.hidden_size])
  x_means_hot, e_loss = vq_nearest_neighbor(
      x, hparams)
  
  if hparams.bottleneck_kind == "mog":
    loss = hparams.beta * e_loss
  else:
    tf.logging.info("Using EMA with beta = {}".format(hparams.beta))
    means, ema_means, ema_count = (hparams.means, hparams.ema_means,
                                   hparams.ema_count)
    # Update the ema variables
    updated_ema_count = commons.assign_moving_average(
        ema_count,
        tf.reduce_sum(x_means_hot, axis=0),
        hparams.decay,
        zero_debias=False)

    dw = tf.matmul(x_means_hot, x, transpose_a=True)
    updated_ema_means = commons.assign_moving_average(
        ema_means, dw, hparams.decay, zero_debias=False)
    n = tf.reduce_sum(updated_ema_count, axis=-1, keepdims=True)
    updated_ema_count = (
        (updated_ema_count + hparams.epsilon) /
        (n + bottleneck_size * hparams.epsilon) * n)
    # pylint: disable=g-no-augmented-assignment
    updated_ema_means = updated_ema_means / tf.expand_dims(
        updated_ema_count, axis=-1)
    # pylint: enable=g-no-augmented-assignment
    with tf.control_dependencies([e_loss]):
      # distribution_strategy
      def update_fn(v, value):
        return tf.assign(v, value)
      tower_context = distribution_strategy_context.get_tower_context()
      if tower_context:
        def merge_fn(strategy, v, value):
          value = strategy.reduce(
              tf.VariableAggregation.MEAN, value, v)
          return strategy.update(v, update_fn, value)
        update_means = tower_context.merge_call(merge_fn, means, updated_ema_means)
      else:
        strategy = distribution_strategy_context.get_cross_tower_context()
        update_means = strategy.update(means, update_fn, updated_ema_means)
      with tf.control_dependencies([update_means]):
        loss = hparams.beta * e_loss

  discrete = tf.reshape(x_means_hot, x_shape[:-1] + [bottleneck_size])
  return discrete, loss


def vq_discrete_unbottleneck(x, hparams):
  """Simple undiscretization from vector quantized representation."""
  x_shape = commons.shape_list(x)
  bottleneck_size = 2**hparams.bottleneck_bits
  means = hparams.means
  x_flat = tf.reshape(x, [-1, bottleneck_size])
  result = tf.matmul(x_flat, means)
  result = tf.reshape(result, x_shape[:-1] + [hparams.hidden_size])
  return result


def residual_conv(x, repeat, k, hparams, name, reuse=None):
  """A stack of convolution blocks with residual connections."""
  with tf.variable_scope(name, reuse=reuse):
    dilations_and_kernels = [(1, k) for _ in range(3)]
    for i in range(repeat):
      with tf.variable_scope("repeat_%d" % i):
        y = commons.conv_block(
            commons.layer_norm(x, name="lnorm"),
            hparams.hidden_size,
            dilations_and_kernels,
            padding="SAME",
            name="residual_conv")
        y = tf.nn.dropout(y, 1.0 - hparams.layer_prepostprocess_dropout)
        x += y
    return x


def compress(x, hparams, name):
  """Compress."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = x
    k1 = 3
    k2 = 2
    cur = residual_conv(cur, hparams.num_compress_steps, k1, hparams, "rc")
    for i in range(hparams.num_compress_steps):
      cur = commons.conv_block(
          cur,
          hparams.hidden_size, [(1, k2)],
          strides=k2,
          name="compress_%d" % i)
    return cur


def decompress_step(source, hparams, first_relu, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = commons.shape_list(source)
    multiplier = 2
    kernel = 1
    thicker = commons.conv_block(
        source,
        hparams.hidden_size * multiplier, [(1, kernel)],
        first_relu=first_relu,
        name="decompress_conv")
    return tf.reshape(thicker, [shape[0], shape[1] * 2, hparams.hidden_size])


def encode(x, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def decode_transformer(encoder_output, encoder_decoder_attention_bias, targets,
                       hparams, name):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    decoder_input, decoder_self_bias = (
        transformer.transformer_prepare_decoder(targets, hparams))

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer.transformer_decoder(
        decoder_input, encoder_output, decoder_self_bias,
        encoder_decoder_attention_bias, hparams)
    return decoder_output


def get_latent_pred_loss(latents_pred, latents_discrete_hot, hparams):
  """Latent prediction and loss."""
  latents_logits = tf.layers.dense(
      latents_pred, 2**hparams.bottleneck_bits, name="extra_logits")
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=tf.stop_gradient(latents_discrete_hot), logits=latents_logits)
  return loss


def ae_latent_sample_beam(latents_dense_in, inputs, ed, embed, hparams):
  """Sample from the latent space in the autoencoder."""
  def symbols_to_logits_fn(ids):
    """Go from ids to logits."""
    latents_discrete = tf.pad(ids[:, 1:], [[0, 0], [0, 1]]) # prepare to be right-shifted in 'decode_transformer'
    #latents_discrete = tf.Print(latents_discrete, [tf.shape(latents_discrete), latents_discrete])

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      latents_dense = embed(
          tf.one_hot(latents_discrete, depth=2**hparams.bottleneck_bits))
      latents_pred = decode_transformer(inputs, ed, latents_dense, hparams,
                                        "extra")
      logits = tf.layers.dense(
          latents_pred, 2**hparams.bottleneck_bits, name="extra_logits")
      current_output_position = commons.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :]
    return logits

  initial_ids = tf.zeros([tf.shape(latents_dense_in)[0]], dtype=tf.int32)
  length = tf.shape(latents_dense_in)[1]
  ids, _, _ = beam_search.beam_search(
      symbols_to_logits_fn,
      initial_ids,
      beam_size=1,
      decode_length=length,
      vocab_size=2**hparams.bottleneck_bits,
      alpha=0.0,
      eos_id=-1,
      stop_early=False)

  res = ids[:, 0, :]  # Pick first beam.
  return res[:, 1:]  # Remove the added all-zeros from ids.


def ae_transformer_internal(inputs, targets, hparams, mode, cache=None):
  """Main step used for training."""
  # Encoder.
  inputs, ed = encode(inputs, hparams, "input_enc")

  # Autoencoding.
  losses = {"extra": tf.constant(0.0), "latent_pred": tf.constant(0.0)}

  max_targets_len_from_inputs = tf.concat([inputs, inputs], axis=1)
  targets, _ = commons.pad_to_same_length(
      targets,
      max_targets_len_from_inputs,
      final_length_divisible_by=2**hparams.num_compress_steps)
  targets_c = compress(targets, hparams, "compress")
  if mode != tf.estimator.ModeKeys.PREDICT:
    # Compress and bottleneck.
    latents_discrete_hot, extra_loss = vq_discrete_bottleneck(
        x=targets_c, hparams=hparams)
    latents_dense = vq_discrete_unbottleneck(
        latents_discrete_hot, hparams=hparams)
    latents_dense = targets_c + tf.stop_gradient(latents_dense - targets_c)
    latents_discrete = tf.argmax(latents_discrete_hot, axis=-1)
    tf.summary.histogram("codes", latents_discrete)
    losses["extra"] = extra_loss

    # Extra loss predicting latent code from input.
    latents_pred = decode_transformer(inputs, ed, latents_dense, hparams,
                                      "extra")
    latent_pred_loss = get_latent_pred_loss(latents_pred, latents_discrete_hot,
                                            hparams)
    losses["latent_pred"] = tf.reduce_mean(latent_pred_loss)
  else:
    latent_len = commons.shape_list(targets_c)[1]
    embed = functools.partial(vq_discrete_unbottleneck, hparams=hparams)
    latents_dense = tf.zeros_like(targets_c)
    if cache is None:
      cache = ae_latent_sample_beam(latents_dense, inputs, ed, embed,
                                    hparams)
    cache_hot = tf.one_hot(cache, depth=2**hparams.bottleneck_bits)
    latents_dense = embed(cache_hot)

  # Postprocess
  d = latents_dense
  d = commons.add_timing_signal_1d(d)

  # Decompressing the dense latents
  for i in range(hparams.num_compress_steps):
    j = hparams.num_compress_steps - i - 1
    d = residual_conv(d, 1, 3, hparams, "decompress_rc_%d" % j)
    d = decompress_step(d, hparams, i > 0, "decompress_%d" % j)

  if hparams.shallow_decoder:
    res = tf.layers.conv1d(d, hparams.hidden_size, 3, padding="same", name="decoder")
  else:
    masking = commons.inverse_lin_decay(hparams.mask_startup_steps)
    masking *= commons.inverse_exp_decay(
        hparams.mask_startup_steps // 4)  # Not much at start.
    masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
    tf.summary.scalar('masking', masking)
    if mode == tf.estimator.ModeKeys.PREDICT:
      masking = 1.0
    mask = tf.less(masking,
                   tf.random_uniform(commons.shape_list(targets)[:-1]))
    mask = tf.expand_dims(tf.to_float(mask), -1)

    # targets is always [batch, length, depth]
    targets = mask * targets + (1.0 - mask) * d

    res = decode_transformer(inputs, ed, targets, hparams, "decoder")
    latent_time = tf.less(hparams.mask_startup_steps,
                          tf.to_int32(tf.train.get_global_step()))
    losses["latent_pred"] *= tf.to_float(latent_time)
  return res, losses, cache


class TransformerNAT():
  """Nonautoregressive Transformer from https://arxiv.org/abs/1805.11063."""

  def __init__(self, hparams, mode):
    self._hparams = hparams
    self.mode = mode

    # lookup tables
    means, ema_means, ema_count = init_vq_bottleneck(
        2**self._hparams.bottleneck_bits, self._hparams.hidden_size, 
        self._hparams.bottleneck_kind=='mog')

    self._hparams.means = means
    self._hparams.ema_means = ema_means
    self._hparams.ema_count = ema_count

  def body(self, features):
    inputs = features["inputs"] if "inputs" in features else None
    reuse = "cache_raw" in features
    with tf.variable_scope('body', reuse=reuse):
      res, loss, _ = ae_transformer_internal(
          inputs, features["targets"],
          self._hparams, self.mode, features.get("cache_raw", None))
      return res, loss

  #def prepare_features_for_infer(self, features):
  #  batch_size = self._decode_hparams.batch_size
  #  inputs = tf.zeros([batch_size, 1, 1, self._hparams.hidden_size])
  #  inputs = inputs if "inputs" in features else None
  #  targets = tf.zeros([batch_size, 1, 1, self._hparams.hidden_size])
  #  with tf.variable_scope("transformer_vqvae/body"):
  #    _, _, cache = ae_transformer_internal(
  #        inputs, targets, self._hparams, tf.estimator.ModeKeys.PREDICT)
  #  features["cache_raw"] = cache

  def infer(self, features):
    """Produce predictions from the model."""
    batch_size = commons.shape_list(features["inputs"])[0]
    length = commons.shape_list(features["inputs"])[1]
    target_length = tf.to_int32(2.0 * tf.to_float(length))
    initial_output = tf.zeros((batch_size, target_length, self._hparams.hidden_size), dtype=tf.float32)

    features["targets"] = initial_output
    decoder_outputs, _ = self.body(features)  # pylint: disable=not-callable
    return decoder_outputs


def build_model_fn(hparams):
  def model_fn(features, labels, mode):
    with tf.variable_scope("transformer_vqvae",
        initializer=tf.variance_scaling_initializer(hparams.initializer_gain, mode="fan_avg", distribution="uniform")):
  
      if mode != tf.estimator.ModeKeys.TRAIN:
        for key in hparams.keys():
          if key.endswith("dropout"):
            setattr(hparams, key, 0.0)
  
      with tf.variable_scope("embeddings",
          initializer=tf.random_normal_initializer(0.0, hparams.hidden_size**-0.5)):
        source_embeddings = tf.get_variable(
            "source_embeddings", [len(hparams.source_vocab), hparams.hidden_size], tf.float32)
        if hparams.shared_embedding:
          target_embeddings = source_embeddings
        else:
          target_embeddings = tf.get_variable(
              "target_embeddings", [len(hparams.target_vocab), hparams.hidden_size], tf.float32)
  
      encoder_input_layer = commons.input_layer(source_embeddings, hparams)
      decoder_input_layer = commons.input_layer(target_embeddings, hparams)
      output_layer = tf.layers.Dense(len(hparams.target_vocab), use_bias=False, name="output")
  
      # create model
      x_enc = encoder_input_layer(features["sources"])
      model = TransformerNAT(hparams, mode) 
  
      # decode
      if mode != tf.estimator.ModeKeys.PREDICT:
        x_dec = decoder_input_layer(features["targets"])
  
        decoder_outputs, losses = model.body(features={'inputs': x_enc, 'targets': x_dec})
        logits = output_layer(decoder_outputs)
        predictions = tf.argmax(logits, -1)
        tgt_len = commons.shape_list(features["targets"])[1]
        losses["cross_entropy"] = commons.compute_loss(logits[:,:tgt_len], features["targets"])
  
        # losses
        loss = 0.
        for k, l in losses.items():
          tf.summary.scalar(k, l)
          loss += l
  
      else:
        decoder_outputs = model.infer(features={'inputs': x_enc})
        logits = output_layer(decoder_outputs)
        predictions = tf.argmax(logits, -1)
        loss = None
  
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = commons.get_train_op(loss, hparams)
    else:
      train_op = None
  
    if mode == tf.estimator.ModeKeys.EVAL:
      # Names tensors to use in the printing tensor hook.
      targets = tf.identity(features["targets"], "targets")
      predictions = tf.identity(predictions, "predictions")
  
      bleu_score = bleu.bleu_score(predictions, targets)
      eval_metrics = {"metrics/approx_bleu_score": tf.metrics.mean(bleu_score)}

      # Summaries
      eval_summary_hook = tf.train.SummarySaverHook(
          save_steps=1,
          output_dir= os.path.join(hparams.model_dir, "eval"),
          summary_op=tf.summary.merge_all())
      eval_summary_hooks = [eval_summary_hook]
    else:
      eval_metrics = None
      eval_summary_hooks = None
  
    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=eval_metrics,
        evaluation_hooks=eval_summary_hooks,
        train_op=train_op)
  return model_fn
