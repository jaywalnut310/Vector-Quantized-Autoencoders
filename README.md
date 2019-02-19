# Vector Quantized Autoencoders
Tensorflow implementation of [Theory and Experiments on Vector Quantized Autoencoders](https://arxiv.org/abs/1805.11063).

By modifying configurations, you can use VQVAE instead of soft EM version VQA (modify bottleneck_kind to vq in config.yml)

For more details, please refer the paper or its precedent paper ([Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)).

* Many codes of this repository are drawn from [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.
* As tensor2tensor is too big to understand at a glance, I draw some of their codes as short as possible.

## Training
```sh
# 1. Create log directory
mkdir /path/to/your-log-dir

# 2. (Optional) Copy configs
cp ./config.yml /path/to/your-log-dir

# 3. Run training
python train.py -m /path/to/your-log-dir
```

If you want to change hparams, then you can do it by choosing one of two options.
* modify config.yml
* add arguments as below:
  ```sh
  python train.py -m /path/to/your-log-dir --c hidden_size=512 num_heads=8
  ```


## WMT14-ENDE (Distilled Data)
I got 24.9 BLEU score. It's quite not bad, but still worse than the paper's result.

I trained with configurations as below:
* 4 V100 GPUs
* batch_size: 8192 in config.yml
* Knowledge Distillation from [Transformer of OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts/wmt)
* averages weights of recent checkpoints using avg_checkpoints.py

It took about 13 days to run 1M train steps.

In my experience, I figured more batch_size and number of gpus help to improve performance significantly.

Thus, there is a possiblility to get further improvements with 8 gpus training. 

If you have enough gpus, please let me know the result.

Additionally, if there is any error, mis-implementation or mis-configuration, please let me know :).


### Data Preparation
TK
```sh
python generate_data.py --c \
  source_vocab_file=/path/to/wmt14_ende_distill/wmtende.vocab \
  target_vocab_file=/path/to/wmt14_ende_distill/wmtende.vocab \
  source_train_file=/path/to/wmt14_ende_distill/train.en \
  target_train_file=/path/to/wmt14_ende_distill/train.de \
  source_eval_file=/path/to/wmt14_ende_distill/valid.en \
  target_eval_file=/path/to/wmt14_ende_distill/valid.de \
  record_train_file=/path/to/wmt_14_ende_distill/train.tfrecords \
  record_eval_file=/path/to/wmt_14_ende_distill/valid.tfrecords
```

### Test
TK
```sh
python decode.py \
  --model_dir /path/to/your-log-dir \
  --predict_file /path/to/wmt14_ende_distill/test.en \
  --out_file out.txt

spm_decode \
  --model=/path/to/OpenNMT-tf/scripts/wmt/wmtende.model \
  --input_format=piece < out.txt > out.detok.txt

sh /path/to/OpenNMT-tf/scripts/wmt/get_ende_bleu.sh out.detok.txt
```

Current result:
```
BLEU = 24.89, 57.6/31.2/19.1/12.2 (BP=0.978, ratio=0.978, hyp_len=63093, ref_len=64496)
```
