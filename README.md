This repository contains code and data for the ACL 2019 paper *Sequence Tagging with Contextual and Non-Contextual Subword Representations: A Multilingual Evaluation* by Benjamin Heinzerling and Michael Strube.

https://www.aclweb.org/anthology/P19-1027/

## POS Tagging Experiments

To run the POS tagging experiments described in the paper use a command like the following, replacing the language code with the one for your language of interest:

```
python main.py train \
    --dataset ud_1_2 \
    --lang et \
    --tag upostag \
    --use-char \
    --use-bpe \
    --use-meta-rnn \
    --best-vocab-size \
    --char-emb-dim 50 \
    --char-nhidden 256 \
    --bpe-nhidden 256 \
    --meta-nhidden 256 \
    --dropout 0.2
```

The arguments `--use-char`, `--use-bpe`, `--use-bert` specify whether to use (randomly initialized) character embeddings, (pretrained) BPE embeddings, and/or (multilingual) BERT. As described in the paper, we use a "meta" LSTM to combine all input representations, which is specified with `--use-meta-rnn`. To automatically select the "best" BPE vocabulary size for the given language, set `--best-vocab-size`, or select a vocabulary size manually with `--vocab-size` (supported vocabulary size for each language can be found [here](https://nlp.h-its.org/bpemb/#download)).

Training will take between 10 minutes and several hours, depending on the size of the dataset for the given language. For the above command you should see training finishwith due to early stopping and then see the results on the test set

```
2020-11-12 12:38:08| score acc_0.9316/0.9328
2020-11-12 12:38:08| out/81/e365_acc_0.9328_model.pt
2020-11-12 12:38:08| Early stop after 10 steps
...
2020-11-12 12:38:11| loaded model out/81/e365_acc_0.9328_model.pt
2020-11-12 12:38:12| score acc_0.8996/0.8996
2020-11-12 12:38:12| out/81/acc_0.8996_model.pt
2020-11-12 12:38:12| final score: 0.8996
```
This particular accuracy is higher than the one reported in the paper (82%) for this language (el: Modern Greek), which might be due to newer versions of pytorch and transformers, but I haven't looked into this.

To train on several languages simultaneously (MultiBPEmb experiments in the paper), run a command like this:

```
python main.py train \
    --dataset ud_1_2_lowres_multi \
    --lang multi \
    --tag upostag \
    --use-char \
    --use-bpe \
    --use-meta-rnn \
    --char-emb-dim 100 \
    --char-nhidden 512 \
    --bpe-nhidden 1024 \
    --meta-nhidden 1024 \
    --dropout 0.2 \
    --vocab-size 1000000
```

## Comments on the code

This code was written in 2018 and would look much different if it were written today.
If you want to develop your own sequence tagging models, I recommend using https://github.com/flairNLP/flair instead.
Project structure:

- argparser.py: contains all command line arguments
- data.py: code for loading plain text data and converting it into pytorch tensors
- main.py: entry point, run either `python main.py train` or `python main.py test`
- model.py: a sequence tagging model
- trainer.py: contains all the boiler plate for loading data and model training
- data: datasets in plain text format
- out/: results and model checkpoints will be written in a subdirectory for each run
- out/cache: tensorized datasets will be cached here

## Requirements

`pytorch transformers numpy bpemb joblib conllu boltons`

The first three libraries are well-known, the other ones are uses for:

- bpemb: pretrained subword embeddings
- joblib: for pickling
- conllu: for reading files in .conllu format
- boltons: useful utility functions

After cloning this repository, extract the UD 1.2 data:

```
$ cd data
$ tar xzf ud_1_2.tar.gz
```
