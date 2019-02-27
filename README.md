
# Vector Quantized Autoencoders

## WARNING: README is not written well currently. I'll clean it in a few weeks!

Tensorflow implementation of [Theory and Experiments on Vector Quantized Autoencoders](https://arxiv.org/abs/1805.11063).

By modifying configurations, you can use VQVAE instead of soft EM version VQA (modify bottleneck_kind to vq in [config.yml](config.yml))

For more details, please refer to the paper or its precedent paper ([Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)).

## Notes
* Many codes of this repository are drawn from [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.
* As tensor2tensor is too big to understand at a glance, I draw some of their codes as concise as possible.
* All works are done in TensorFlow 1.12.


## WMT14-ENDE (Distilled Data)
I got 24.9 BLEU score. It's quite not that bad, but still worse than the paper's result.

I trained with configurations as below:
* 4 V100 GPUs
* batch_size: 8192
* Knowledge Distillation from [Transformer of OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts/wmt)
* averages weights of recent checkpoints using [avg_checkpoints.py](avg_checkpoints.py)

It took about 13 days to run 1M train steps.

In my experience, I figured more batch_size and number of gpus help to improve performance significantly.

Thus, there is a possiblility to get further improvements with 8 or more gpus training.

If you have enough gpus, please let me know the result.

Additionally, if there is any error, mis-implementation or mis-configuration, please let me know :).


### Data Preparation
TK
```sh
# You can change filepaths by modifying 'config.yml'
# or you can change filepaths using --c config as below.
# python generate_data.py --c \
#   source_vocab_file=/path/to/your-data-dir/vocab_src \
#   target_vocab_file=/path/to/your-data-dir/vocab_tgt \
#   source_train_file=/path/to/your-data-dir/train_src \
#   target_train_file=/path/to/your-data-dir/train_tgt \
#   source_eval_file=/path/to/your-data-dir/valid_src \
#   target_eval_file=/path/to/your-data-dir/valid_tgt \
#   record_train_file=/path/to/your-data-dir/train.tfrecords \
#   record_eval_file=/path/to/your-data-dir/valid.tfrecords
python generate_data.py
```

### Training
```sh
# 1. Create log directory
mkdir /path/to/your-log-dir

# 2. (Optional) Copy configs
cp ./config.yml /path/to/your-log-dir

# 3. Run training
python train.py -m /path/to/your-log-dir
```

If you want to change hparams, then you can do it by choosing one of two options.
* modify [config.yml](config.yml)
* add arguments as below:
  ```sh
  python train.py -m /path/to/your-log-dir --c hidden_size=512 num_heads=8
  ```

### Test
TK
```sh
# (Optional) Averaging checkpoints is mostly helpful to improve performance
python avg_checkpoints.py --prefix=/path/to/your-log-dir --num_last_checkpoints=20

# checkpoint config is optional
python decode.py \
  --model_dir /path/to/your-log-dir \
  --predict_file /path/to/wmt14_ende_distill/test.en \
  --out_file out.txt
  --checkpoint /tmp/averaged.ckpt-0

spm_decode \
  --model=/path/to/OpenNMT-tf/scripts/wmt/wmtende.model \
  --input_format=piece < out.txt > out.detok.txt

sh /path/to/OpenNMT-tf/scripts/wmt/get_ende_bleu.sh out.detok.txt
```

Current result:
```
BLEU = 24.89, 57.6/31.2/19.1/12.2 (BP=0.978, ratio=0.978, hyp_len=63093, ref_len=64496)
```

#### Prediction samples
| Source        | Prediction    | Ground Truth  |
| ------------- | ------------- | ------------- |
| Gutach: Increased safety for pedestrians | Guts: Mehr Sicherheit für Fußgänger | Gutach: Noch mehr Sicherheit für Fußgänger |
| They are not even 100 metres apart: On Tuesday, the new B 33 pedestrian lights in Dorfparkplatz in Gutach became operational - within view of the existing Town Hall traffic lights. | Sie sind nicht nicht einmal 100 Meter voneinander entfernt: Am Dienstag wurden die neuen Fußgängerzonen B 33 am Dorfparkplatz in Gutach im Hinblick auf die bestehende Ampel des Rathauses in Betrieb genommen. | Sie stehen keine 100 Meter voneinander entfernt: Am Dienstag ist in Gutach die neue B 33-Fußgängerampel am Dorfparkplatz in Betrieb genommen worden - in Sichtweite der älteren Rathausampel. |
| Two sets of lights so close to one another: intentional or just a silly error? | Zwei Lichtsätze so nah aneinander: absichtlich oder nur ein dummer Fehler? | Zwei Anlagen so nah beieinander: Absicht oder Schildbürgerstreich? |

The above samples are just the first 3 sentences in the test set.
More samples can be found in [source.txt](resources/source.txt), [prediction.txt](resources/prediction.txt) and [ground_truth.txt](resources/ground_truth.txt).

