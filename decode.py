import argparse

import tensorflow as tf

from hparams import create_hparams
import data
import utils
import transformer_vq as model


def main():
    parser = utils.get_argument_parser("Decode by using the trained model")
    parser.add_argument("--checkpoint", dest="checkpoint", help="Path to a checkpoint file. Default is the latest version.")
    parser.add_argument("--limit", type=int, default=0, help="The number of sentences to be decoded. (default=unlimited)")
    parser.add_argument("--use_eval", action="store_true", help="Use evaluation dataset for prediction")
    parser.add_argument("--predict_file", type=str, default="", help="Path to a text file to be translated")
    parser.add_argument("--out_file", type=str, default="", help="Path to a text file to write")
    args = parser.parse_args()

    hparams = create_hparams(args.model_dir, args.configs, initialize=False)
    utils.check_git_hash(args.model_dir)

    data.load_vocab(hparams)
    if args.use_eval:
        pipeline = data.InputPipeline(hparams.source_eval_file, None, None, tf.estimator.ModeKeys.PREDICT, hparams)
    else:
        pipeline = data.InputPipeline(args.predict_file, None, None, tf.estimator.ModeKeys.PREDICT, hparams)

    estimator = tf.estimator.Estimator(model_fn=model.build_model_fn(hparams), model_dir=args.model_dir)

    # set a file path to write
    if args.out_file != "":
        f = open(args.out_file, 'w')
    else:
        f = None

    for i, prediction in enumerate(estimator.predict(pipeline, checkpoint_path=args.checkpoint)):
        if args.limit and i == args.limit:
            break
        token_ids = prediction.tolist()
        print(hparams.target_vocab.decode(token_ids), file=f)
        if i % 1000 == 0:
            tf.logging.info("write: %d", i)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
