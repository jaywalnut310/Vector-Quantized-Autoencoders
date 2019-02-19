import tensorflow as tf
from tensorflow.python.training import distribution_strategy_context


"""
Minimized tensor2tensor utils.
Almost codes are drawn from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor
"""

#== UTILS ==

def shape_list(x):
  """Shape list"""
  x_shape = tf.shape(x)
  x_get_shape = x.get_shape().as_list()

  res = []
  for i, d in enumerate(x_get_shape):
    if d is not None:
      res.append(d)
    else:
      res.append(x_shape[i])
  return res


def should_generate_summaries():
  """Is this an appropriate context to generate summaries.
  Returns:
    a boolean
  """
  name_scope = tf.contrib.framework.get_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True


def log_prob_from_logits(logits, reduce_axis=-1):
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


def pad_to_same_length(x, y, final_length_divisible_by=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[1]
    y_length = shape_list(y)[1]

    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by

    res_x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
    res_y = tf.pad(y, [[0, 0], [0, max_length - y_length], [0, 0]])
    return res_x, res_y


def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.
  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
    embedding vector is all zero, and is 0 otherwise.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.0))


def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.
  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].
  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.
  A position may attend to positions at most max_distance from it,
  forward and backwards.
  This does not actually save any computation.
  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = common_layers.ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.
  Allows a query to attend to all positions up to and including its own.
  Args:
   length: a Scalar.
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = tf.matrix_band_part(
      tf.ones([length, length]), -1, 0)
  ret = -1e9 * (1.0 - band)
  return tf.reshape(ret, [1, 1, length, length])


def add_timing_signal_1d(x):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Args:
    x: a Tensor with shape [batch, length, channels]
  Returns:
    a Tensor the same shape as x.
  """
  with tf.name_scope("positional_encoding"):
    length, channels = shape_list(x)[1:3]
    num_timescales = tf.to_float(channels // 2)
    log_timescale_increment = tf.log(10000.) / tf.maximum(tf.to_float(num_timescales) - 1, 1)

    position = tf.to_float(tf.range(length)) # [L]
    inv_timescales = tf.exp(tf.range(num_timescales) * -log_timescale_increment) # [D/2]
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0) # [L, D/2]
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], 1) # [L, D]
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels]) # [1, L, D]

    return x + signal


def inverse_exp_decay(max_step, min_value=0.01):
  """Inverse-decay exponentially from 0.01 to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  step = tf.to_float(tf.train.get_global_step())
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01):
  """Inverse-decay linearly from 0.01 to 1.0 reached at max_step."""
  step = tf.to_float(tf.train.get_global_step())
  progress = tf.minimum(step / float(max_step), 1.0)
  return progress * (1.0 - min_value) + min_value


def compute_loss(logits, labels, lengths=None):
  with tf.name_scope("compute_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)

    # Mask out the losses we don't care about
    if lengths is None:
      lengths = sequence_length(labels)
    mask = tf.sequence_mask(lengths, tf.shape(labels)[1])
    losses = losses * tf.to_float(mask)
    loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(lengths))

    return loss


def input_layer(embeddings, hparams):
  def input_layer_fn(inputs, name=None):
    """Common part for encoder and decoder"""
    with tf.variable_scope(name, "input"):
      embedded = tf.nn.embedding_lookup(embeddings, inputs)

      # scaling & add positional encoding
      scaled = embedded * hparams.hidden_size**0.5

      # mask out padding
      padding_mask = tf.cast(tf.cast(inputs, tf.bool), tf.int32)
      outputs = scaled * tf.expand_dims(tf.to_float(padding_mask), -1)

      return outputs
  return input_layer_fn


def sequence_length(x):
  """Count non-zero values

  Arguments:
    x: [B, L]

  Returns:
    [B]
  """
  x = tf.cast(x, tf.bool)
  x = tf.cast(x, tf.int32)
  return tf.reduce_sum(x, -1)


def prepend_token(x, token):
  """Prepend a token to x

  Arguments:
    x: [B, L]
    token: int32 scalar

  Returns:
    [B, L]
  """
  batch_size = tf.shape(x)[0]
  tokens = tf.fill([batch_size, 1], token)
  x = tf.concat([tokens, x], -1)[:, :-1]
  return x


def make_initial_tokens(batch_size, token):
  """Make intial tokens for decoder"""
  x = tf.constant([[token]], dtype=tf.int32)
  tokens = tf.tile(x, [batch_size, 1])
  return tokens


def get_optimizer_fn(hparams):
  if hparams.optimizer == "adam":
    return lambda lr: tf.train.AdamOptimizer(
          lr, hparams.adam_beta1, hparams.adam_beta2, hparams.adam_epsilon)

  else:
    raise ValueError("Unknown optimizer: {}".format(hparams.optimizer))


def noam_learning_rate_decay(learning_rate, global_step, hparams):
  step = tf.to_float(global_step)
  return learning_rate * hparams.hidden_size**-0.5 * tf.minimum(
      (step + 1) * hparams.lr_warmup_steps**-1.5, (step + 1)**-0.5)


def get_learning_rate_decay_fn(hparams):
  if hparams.lr_decay is None:
    return lambda lr, step: lr

  elif hparams.lr_decay == "noam":
    return lambda lr, step: noam_learning_rate_decay(lr, step, hparams)

  else:
    raise ValueError("Unknown learing rate decay method: {}".format(hparams.lr_decay))


def get_train_op(loss, hparams, name="train"):
  optimizer_fn = get_optimizer_fn(hparams)

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=None,
      learning_rate=hparams.learning_rate,
      learning_rate_decay_fn=get_learning_rate_decay_fn(hparams),
      optimizer=get_optimizer_fn(hparams),
      clip_gradients=hparams.clip_gradients,
      summaries=None,
      name=name)

  return train_op


def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
  """Compute the moving average of a variable.
  https://github.com/tensorflow/tensorflow/blob/c966b5eed60a570f2121cb84ddb4ece84c413719/tensorflow/python/training/moving_averages.py
  """

  def _zero_debias(unbiased_var, value, decay):
    """Compute the delta required for a debiased Variable.
    """
    with tf.variable_scope(
        unbiased_var.op.name, values=[unbiased_var, value, decay]) as scope:
      with tf.colocate_with(unbiased_var):
        with tf.init_scope():
          biased_initializer = tf.zeros_initializer(
              dtype=unbiased_var.dtype)(unbiased_var.get_shape())
          local_step_initializer = tf.zeros_initializer()
        def _maybe_get_unique(name):
          """Get name for a unique variable, if not `reuse=True`."""
          if tf.get_variable_scope().reuse:
            return name
          vs_vars = [x.op.name for x in
                     tf.get_variable_scope().global_variables()]
          full_name = tf.get_variable_scope().name + "/" + name
          if full_name not in vs_vars: return name
          idx = 1
          while full_name + ("_%d" % idx) in vs_vars:
            idx += 1
          return name + ("_%d" % idx)
        biased_var = tf.get_variable(
            _maybe_get_unique("biased"), initializer=biased_initializer,
            trainable=False)
        local_step = tf.get_variable(
            _maybe_get_unique("local_step"),
            shape=[],
            dtype=unbiased_var.dtype,
            initializer=local_step_initializer,
            trainable=False)

        # Get an update ops for both shadow variables.
        update_biased = tf.assign_sub(biased_var,
                                             (biased_var - value) * decay,
                                             name=scope.name)
        update_local_step = local_step.assign_add(1)

        # Compute the value of the delta to update the unbiased EMA. Make sure to
        # use the new values of the biased variable and the local step.
        with tf.control_dependencies([update_biased, update_local_step]):
          # This function gets `1 - decay`, so use `1.0 - decay` in the exponent.
          unbiased_ema_delta = (unbiased_var - biased_var.read_value() /
                                (1 - tf.pow(
                                    1.0 - decay, local_step.read_value())))

        return unbiased_ema_delta

  def update_fn(v, value, decay=decay):
    decay = tf.convert_to_tensor(1.0 - decay, name="decay")
    if decay.dtype != v.dtype.base_dtype:
      decay = tf.cast(decay, v.dtype.base_dtype)
    if zero_debias:
      update_delta = _zero_debias(v, value, decay)
    else:
      update_delta = (v - value) * decay
    return tf.assign_sub(v, update_delta, name=scope)

  with tf.name_scope(name, "AssignMovingAvg",
                      [variable, value, decay]) as scope:
    tower_context = distribution_strategy_context.get_tower_context()
    if tower_context:
      # In a tower context, we update variable using the mean of value across
      # towers.
      def merge_fn(strategy, v, value):
        value = strategy.reduce(
            tf.VariableAggregation.MEAN, value, v)
        return strategy.update(v, update_fn, value)

      return tower_context.merge_call(merge_fn, variable, value)
    else:
      strategy = distribution_strategy_context.get_cross_tower_context()
      return strategy.update(variable, update_fn, value)


#== LAYERS ==

def layer_norm(x, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    filters = shape_list(x)[-1]
    scale = tf.get_variable("scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable("bias", [filters], initializer=tf.zeros_initializer())

    # Layer norm computation
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         epsilon,
                         default_name,
                         name=None):
  """Apply a sequence of functions to the input or output of a layer.
  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout
  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))
  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
  Returns:
    a Tensor
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "n":
        x = layer_norm(x, epsilon)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        x = tf.nn.dropout(x, 1.0 - dropout_rate)
    return x


def layer_preprocess(layer_input, hparams):
  """Apply layer preprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_epsilon
  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.
  Returns:
    a Tensor
  """
  assert "a" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=hparams.layer_preprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      epsilon=hparams.norm_epsilon,
      default_name="layer_preprocess")


def layer_postprocess(layer_input, layer_output, hparams):
  """Apply layer postprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_epsilon
  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.
  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      epsilon=hparams.norm_epsilon,
      default_name="layer_postprocess")


def conv_block(inputs,
               filters,
               dilation_rates_and_kernel_sizes,
               strides=1,
               padding='VALID',
               first_relu=True,
               name=None):
  """A block of convolutions.
  Args:
    inputs: a Tensor
    filters: an Integer
    dilation_rates_and_kernel_sizes: a list of tuples (dilation, (k_w, k_h))
    first_relu: whether to do a relu at start (defaults to True)
  Returns:
     a Tensor.
  """

  with tf.variable_scope(name, "conv_block", [inputs]):
    cur, counter = inputs, -1
    for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
      counter += 1
      if first_relu or counter > 0:
        cur = tf.nn.relu(cur)
      cur = tf.layers.conv1d(
          cur,
          filters,
          kernel_size,
          strides,
          padding=padding,
          dilation_rate=dilation_rate,
          name="conv_block_%d" % counter,
          use_bias=False)
      cur = layer_norm(cur, name="conv_block_norm_%d" % counter)
    return cur


#== ATTENTION MODULES ==

def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  x_new = tf.transpose(x, [0, 2, 1, 3])
  x_shape = shape_list(x_new)
  a, b = x_shape[-2:]
  return tf.reshape(x_new, x_shape[:-2] + [a * b])


def compute_attention_component(antecedent,
                       total_depth,
                       filter_width=1,
                       padding="VALID",
                       name="c"):
  """Computes attention compoenent (query, key or value).
  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
 
  Returns:
    c : [batch, length, depth] tensor
  """
  if filter_width == 1:
    return tf.layers.dense(antecedent, total_depth, use_bias=False, name=name)
  else:
    return tf.layers.conv1d(antecedent, total_depth, filter_width, padding=padding, name=name)


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
  """Computes query, key and value.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent, total_key_depth, q_filter_width, q_padding, "q")
  k = compute_attention_component(
      memory_antecedent, total_key_depth, kv_filter_width, kv_padding, "k")
  v = compute_attention_component(
      memory_antecedent, total_value_depth, kv_filter_width, kv_padding, "v")
  return q, k, v


def attention_image_summary(attn):
  """Compute color image summary.
  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
  """
  attn = tf.cast(attn, tf.float32)
  num_heads = shape_list(attn)[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  tf.summary.image("attention", image, max_outputs=1)


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name=None,
                          make_image_summary=True):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      bias = tf.cast(bias, logits.dtype)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if should_generate_summaries() and make_image_summary:
      attention_image_summary(weights)
    return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        name="multihead_attention",
                        make_image_summary=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    **kwargs (dict): Parameters for the attention function
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hiddem_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      values=[query_antecedent, memory_antecedent]):

    if cache is None or memory_antecedent is None:
      q, k, v =  compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                            total_value_depth, q_filter_width, kv_filter_width,
                            q_padding, kv_padding)
    if cache is not None:
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q")
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        k = cache["k"] = tf.concat([cache["k"], k], axis=2)
        v = cache["v"] = tf.concat([cache["v"], v], axis=2)

    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    x = dot_product_attention(q, k, v, bias, dropout_rate,
                              make_image_summary=make_image_summary)
    x = combine_heads(x)

    # Set last dim specifically.
    tf.reshape(x, shape_list(x)[:-1] + [total_value_depth])

    x = tf.layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x
