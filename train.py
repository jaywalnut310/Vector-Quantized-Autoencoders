import tensorflow as tf

from hparams import create_hparams
import data
import utils
import transformer_vq as model


def tensors_to_string(hparams, N=5):
  def transform(tensors):
    targets = tensors["targets"][:N].tolist()
    predictions = tensors["predictions"][:N].tolist()
    result = []
    for t, p in zip(targets, predictions):
      result.append("")
      result.append(hparams.target_vocab.decode(t))
      result.append(hparams.target_vocab.decode(p))
    return "\n".join(result)
  return transform


def main(argv):
    args = utils.parse_args("Train a transformer model")
    utils.redirect_log_to_file(args.model_dir)

    hparams = create_hparams(args.model_dir, args.configs, initialize=True)
    utils.check_git_hash(args.model_dir)

    # Prepare data
    data.load_vocab(hparams)

    train_input_fn = data.InputPipeline(
            None,
            None,
            hparams.record_train_file,
            tf.estimator.ModeKeys.TRAIN,
            hparams)
    eval_input_fn = data.InputPipeline(
            None,
            None,
            hparams.record_eval_file,
            tf.estimator.ModeKeys.EVAL,
            hparams)

    # Training
    log_samples_hook = tf.train.LoggingTensorHook(
            ['targets', 'predictions'], at_end=True, formatter=tensors_to_string(hparams))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
            max_steps=hparams.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
            steps=hparams.eval_steps,
            hooks=[log_samples_hook])

    distribution = tf.contrib.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(model_dir=args.model_dir,
            train_distribute=distribution,
            save_summary_steps=hparams.save_summary_steps,
            save_checkpoints_steps=hparams.save_checkpoints_steps,
            keep_checkpoint_max=hparams.n_checkpoints)
    estimator = tf.estimator.Estimator(
            model_fn=model.build_model_fn(hparams),
            config=run_config,
            model_dir=args.model_dir)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
