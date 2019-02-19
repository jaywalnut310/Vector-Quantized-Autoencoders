import math
import tensorflow as tf

import commons
import bleu


def transformer_prepare_encoder(inputs, hparams):
  """Prepare one shard of the model for the encoder.
  Args:
    inputs: a Tensor.
    hparams: run hyperparameters
  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  encoder_input = inputs
  encoder_padding = commons.embedding_to_padding(encoder_input)
  ignore_padding = commons.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  encoder_input = commons.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        make_image_summary=True):
  """A stack of transformer layers.
  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
    hparams: hyperparameters for model
    make_image_summary: Whether to make an attention image summary.
  Returns:
    y: a Tensors
  """
  x = encoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_encoder_layers or
                       hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = commons.multihead_attention(
              commons.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              make_image_summary=make_image_summary)
          x = commons.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              commons.layer_preprocess(x, hparams), hparams)
          x = commons.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return commons.layer_preprocess(x, hparams)


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.
  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.
  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  decoder_self_attention_bias = (
      commons.attention_bias_lower_triangle(
          commons.shape_list(targets)[1]))

  decoder_input = commons.shift_right_3d(targets)
  decoder_input = commons.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder",
                        make_image_summary=True):
  """A stack of transformer layers.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
    make_image_summary: Whether to make an attention image summary.
  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or
                       hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = commons.multihead_attention(
              commons.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              cache=layer_cache,
              make_image_summary=make_image_summary)
          x = commons.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = commons.multihead_attention(
                commons.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                cache=layer_cache,
                make_image_summary=make_image_summary)
            x = commons.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              commons.layer_preprocess(x, hparams), hparams)
          x = commons.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return commons.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams):
  """Feed-forward layer in the transformer.
  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  x = tf.layers.dense(x, hparams.filter_size, activation=tf.nn.relu, name="conv1")
  if hparams.relu_dropout != 0.0:
    x = tf.nn.dropout(x, 1.0 - hparams.relu_dropout)
  x = tf.layers.dense(x, hparams.hidden_size, name="conv2")
  return x
